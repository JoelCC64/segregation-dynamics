!===============================================================================
! segregation.f90 — Continuous-space Schelling-type segregation model
!
! Reference forward-simulation implementation in standard Fortran 95.
! Part of the "segregation-dynamics" project: a fast compiled forward model
! (this file) alongside a differentiable PyTorch re-implementation (/pytorch)
! for gradient-based calibration and sensitivity analysis.
!
! MODEL
!   N_A + N_B point agents of two types live in the square [0,L) x [0,L)
!   (open boundaries, no periodicity). An agent is "happy" if it has at
!   least K_MIN neighbours of its own type within Euclidean distance R.
!   Dynamics: at each event, one uniformly random unhappy agent teleports
!   to uniformly random locations (up to MAX_TRIES trials) and settles at
!   the first location where it is happy; if none is found, it remains at
!   its last trial location. The run stops when every agent is happy or
!   after MAX_MOVES events.
!
! IMPLEMENTATION
!   Neighbour search uses a cell-linked list on an M x M grid; the 3x3
!   block search is exact because the cell size L/M is >= R (checked at
!   startup). The unhappy set is kept exactly up to date: after each event
!   only agents near the departure and arrival points are re-evaluated,
!   since a teleport changes neighbour counts only within radius R of
!   those two points.
!
! BUILD:   make                (or: gfortran -O2 -std=f95 -o segregation segregation.f90)
! RUN:     ./segregation
! OUTPUT:  metrics.dat, snapshots.dat, final_configuration.dat
!===============================================================================
program segregation
    implicit none

    ! ----- numeric precision --------------------------------------------------
    integer, parameter :: wp = selected_real_kind(12)

    ! ----- model parameters ----------------------------------------------------
    integer,  parameter :: N_A = 1000          ! agents of type 1
    integer,  parameter :: N_B = 1000          ! agents of type 0
    integer,  parameter :: N   = N_A + N_B
    real(wp), parameter :: L   = 100.0_wp      ! side of the square domain
    real(wp), parameter :: R   = 10.0_wp       ! neighbourhood radius
    integer,  parameter :: K_MIN = 30          ! same-type neighbours required
    integer,  parameter :: MAX_MOVES = 10000   ! relocation-event budget
    integer,  parameter :: MAX_TRIES = 40      ! trial locations per event
    integer,  parameter :: SEED = 20260707     ! RNG seed (reproducible runs)

    ! ----- output cadence -------------------------------------------------------
    integer, parameter :: METRICS_EVERY  = 100   ! events between metric samples
    integer, parameter :: SNAPSHOT_EVERY = 1000  ! events between snapshots

    ! ----- cell grid for neighbour search ----------------------------------------
    integer,  parameter :: M    = 10           ! cells per side
    real(wp), parameter :: CELL = L / M        ! cell size; must satisfy CELL >= R

    ! ----- state ------------------------------------------------------------------
    real(wp) :: x(N), y(N)
    integer  :: atype(N)                       ! agent type: 1 or 0
    logical  :: happy(N)

    integer :: head(M, M)                      ! first agent in each cell (0 = empty)
    integer :: nxt(N), prv(N)                  ! doubly linked list within each cell
    integer :: cx_of(N), cy_of(N)              ! current cell of each agent

    integer :: unhappy_list(N)                 ! entries 1..n_unhappy: unhappy agents
    integer :: list_pos(N)                     ! inverse map into unhappy_list (0 = happy)
    integer :: n_unhappy

    ! ----- bookkeeping ---------------------------------------------------------------
    integer  :: a, event, trial, events_done, total_trials, failed_events
    real(wp) :: x_old, y_old, rn
    real(wp) :: frac_happy, seg_index
    logical  :: settled

    call seed_rng(SEED)

    ! The 3x3 block search silently misses neighbours if cells are smaller
    ! than the interaction radius; refuse to run in that case.
    if (CELL < R) then
        print *, 'ERROR: cell size L/M < radius R; 3x3 cell search would miss neighbours.'
        stop 1
    end if

    call initialise_agents()
    call rebuild_cell_lists()
    call evaluate_all()

    open(unit=20, file='metrics.dat',             status='replace', action='write')
    open(unit=21, file='snapshots.dat',           status='replace', action='write')
    open(unit=22, file='final_configuration.dat', status='replace', action='write')

    write(20, '(A)') '#    event   n_unhappy   frac_happy    seg_index'
    call report_metrics(0)
    call write_snapshot(21, 0)

    events_done   = 0
    total_trials  = 0
    failed_events = 0

    do event = 1, MAX_MOVES
        if (n_unhappy == 0) exit               ! convergence: everyone is happy

        ! Pick a uniformly random unhappy agent: rn in [0,1) -> index in 1..n_unhappy.
        call random_number(rn)
        a = unhappy_list(int(rn * n_unhappy) + 1)

        x_old = x(a)
        y_old = y(a)

        ! Teleport to random locations until happy; keep the last trial
        ! location if no happy spot is found within MAX_TRIES.
        settled = .false.
        do trial = 1, MAX_TRIES
            call random_number(rn)
            x(a) = rn * L
            call random_number(rn)
            y(a) = rn * L
            call update_cell_membership(a)
            total_trials = total_trials + 1
            if (is_happy(a)) then
                settled = .true.
                exit
            end if
        end do
        if (.not. settled) failed_events = failed_events + 1

        ! Only agents within R of the departure or arrival point can have a
        ! changed neighbour count; re-evaluate exactly those regions (the
        ! mover itself lies inside the arrival block).
        call refresh_region(x_old, y_old)
        call refresh_region(x(a), y(a))

        events_done = events_done + 1
        if (mod(event, METRICS_EVERY)  == 0) call report_metrics(event)
        if (mod(event, SNAPSHOT_EVERY) == 0) call write_snapshot(21, event)
    end do

    if (mod(events_done, METRICS_EVERY) /= 0) call report_metrics(events_done)
    call write_snapshot(22, events_done)

    call compute_metrics(frac_happy, seg_index)
    if (n_unhappy == 0) then
        print '(A,I0,A)', 'Converged: all agents happy after ', events_done, ' relocation events.'
    else
        print '(A,I0,A,I0,A)', 'Stopped after ', events_done, ' events; ', n_unhappy, &
                               ' agents still unhappy.'
    end if
    print '(A,F6.3)', 'fraction happy                    = ', frac_happy
    print '(A,F6.3)', 'mean same-type neighbour fraction = ', seg_index
    print '(A,I0,A,I0,A)', 'trial moves = ', total_trials, '  (', failed_events, &
                           ' events found no happy spot)'

    close(20)
    close(21)
    close(22)

contains

    !---------------------------------------------------------------------------
    ! Deterministic, portable seeding of the intrinsic RNG: random_seed()
    ! with no arguments is compiler-dependent and not reproducible.
    subroutine seed_rng(base)
        integer, intent(in) :: base
        integer :: nseed, i
        integer, allocatable :: s(:)
        call random_seed(size=nseed)
        allocate(s(nseed))
        do i = 1, nseed
            s(i) = base + 37 * (i - 1)
        end do
        call random_seed(put=s)
        deallocate(s)
    end subroutine seed_rng

    !---------------------------------------------------------------------------
    ! Uniform random positions in [0,L)^2; first N_A agents are type 1.
    subroutine initialise_agents()
        integer  :: i
        real(wp) :: r1
        do i = 1, N
            call random_number(r1)
            x(i) = r1 * L
            call random_number(r1)
            y(i) = r1 * L
            if (i <= N_A) then
                atype(i) = 1
            else
                atype(i) = 0
            end if
        end do
        happy(:)    = .true.
        list_pos(:) = 0
        n_unhappy   = 0
    end subroutine initialise_agents

    !---------------------------------------------------------------------------
    ! Cell index from position by direct arithmetic. Clamping makes the
    ! edge case p == L safe.
    subroutine locate_cell(px, py, cx, cy)
        real(wp), intent(in)  :: px, py
        integer,  intent(out) :: cx, cy
        cx = max(1, min(M, int(px / CELL) + 1))
        cy = max(1, min(M, int(py / CELL) + 1))
    end subroutine locate_cell

    !---------------------------------------------------------------------------
    ! Cell-linked lists: head(cx,cy) points to the first agent in the cell and
    ! nxt/prv chain the rest, giving O(1) insertion and removal (the standard
    ! molecular-dynamics structure).
    subroutine rebuild_cell_lists()
        integer :: i
        head(:, :) = 0
        do i = 1, N
            call insert_into_cell(i)
        end do
    end subroutine rebuild_cell_lists

    subroutine insert_into_cell(i)
        integer, intent(in) :: i
        integer :: cx, cy
        call locate_cell(x(i), y(i), cx, cy)
        cx_of(i) = cx
        cy_of(i) = cy
        nxt(i)   = head(cx, cy)
        prv(i)   = 0
        if (head(cx, cy) /= 0) prv(head(cx, cy)) = i
        head(cx, cy) = i
    end subroutine insert_into_cell

    subroutine remove_from_cell(i)
        integer, intent(in) :: i
        if (prv(i) == 0) then
            head(cx_of(i), cy_of(i)) = nxt(i)
        else
            nxt(prv(i)) = nxt(i)
        end if
        if (nxt(i) /= 0) prv(nxt(i)) = prv(i)
    end subroutine remove_from_cell

    ! Call after changing x(i), y(i).
    subroutine update_cell_membership(i)
        integer, intent(in) :: i
        call remove_from_cell(i)
        call insert_into_cell(i)
    end subroutine update_cell_membership

    !---------------------------------------------------------------------------
    ! Same-type and total neighbour counts within radius R of agent i, using
    ! the 3x3 block of cells around its own cell (exact because CELL >= R).
    subroutine count_neighbours(i, n_same, n_total)
        integer, intent(in)  :: i
        integer, intent(out) :: n_same, n_total
        integer  :: jx, jy, b
        real(wp) :: d2
        n_same  = 0
        n_total = 0
        do jy = max(1, cy_of(i) - 1), min(M, cy_of(i) + 1)
            do jx = max(1, cx_of(i) - 1), min(M, cx_of(i) + 1)
                b = head(jx, jy)
                do while (b /= 0)
                    if (b /= i) then
                        d2 = (x(b) - x(i))**2 + (y(b) - y(i))**2
                        if (d2 <= R * R) then
                            n_total = n_total + 1
                            if (atype(b) == atype(i)) n_same = n_same + 1
                        end if
                    end if
                    b = nxt(b)
                end do
            end do
        end do
    end subroutine count_neighbours

    logical function is_happy(i)
        integer, intent(in) :: i
        integer :: n_same, n_total
        call count_neighbours(i, n_same, n_total)
        is_happy = (n_same >= K_MIN)
    end function is_happy

    !---------------------------------------------------------------------------
    ! Exact bookkeeping of the unhappy set with O(1) insert/remove via an
    ! inverse index map.
    subroutine set_happiness(i, h)
        integer, intent(in) :: i
        logical, intent(in) :: h
        integer :: p, last
        if (h .neqv. happy(i)) then
            if (h) then                          ! unhappy -> happy: remove
                p    = list_pos(i)
                last = unhappy_list(n_unhappy)
                unhappy_list(p) = last
                list_pos(last)  = p
                n_unhappy       = n_unhappy - 1
                list_pos(i)     = 0
            else                                 ! happy -> unhappy: append
                n_unhappy               = n_unhappy + 1
                unhappy_list(n_unhappy) = i
                list_pos(i)             = n_unhappy
            end if
            happy(i) = h
        end if
    end subroutine set_happiness

    subroutine evaluate_all()
        integer :: i
        do i = 1, N
            call set_happiness(i, is_happy(i))
        end do
    end subroutine evaluate_all

    !---------------------------------------------------------------------------
    ! Re-evaluate every agent in the 3x3 cell block around (px,py). Any agent
    ! whose neighbour count changed after a teleport lies within R of the
    ! departure or arrival point, hence inside one of these two blocks.
    subroutine refresh_region(px, py)
        real(wp), intent(in) :: px, py
        integer :: cx, cy, jx, jy, b
        call locate_cell(px, py, cx, cy)
        do jy = max(1, cy - 1), min(M, cy + 1)
            do jx = max(1, cx - 1), min(M, cx + 1)
                b = head(jx, jy)
                do while (b /= 0)
                    call set_happiness(b, is_happy(b))
                    b = nxt(b)
                end do
            end do
        end do
    end subroutine refresh_region

    !---------------------------------------------------------------------------
    ! Global observables:
    !   frac_happy — fraction of happy agents (exact, from the unhappy set).
    !   seg_index  — mean, over agents with at least one neighbour, of the
    !                fraction of same-type agents among their neighbours
    !                within R:  0.5 = perfectly mixed, -> 1.0 = segregated.
    subroutine compute_metrics(fh, seg)
        real(wp), intent(out) :: fh, seg
        integer  :: i, n_same, n_total, n_counted
        real(wp) :: acc
        acc       = 0.0_wp
        n_counted = 0
        do i = 1, N
            call count_neighbours(i, n_same, n_total)
            if (n_total > 0) then
                acc       = acc + real(n_same, wp) / real(n_total, wp)
                n_counted = n_counted + 1
            end if
        end do
        seg = acc / real(max(n_counted, 1), wp)
        fh  = real(N - n_unhappy, wp) / real(N, wp)
    end subroutine compute_metrics

    subroutine report_metrics(event_no)
        integer, intent(in) :: event_no
        real(wp) :: fh, seg
        call compute_metrics(fh, seg)
        write(20, '(I10, I12, 2F13.6)') event_no, n_unhappy, fh, seg
    end subroutine report_metrics

    !---------------------------------------------------------------------------
    ! One "x y type" block per snapshot, separated by two blank lines
    ! (gnuplot 'index' convention; also easy to parse from Python).
    subroutine write_snapshot(unit_no, event_no)
        integer, intent(in) :: unit_no, event_no
        integer :: i
        write(unit_no, '(A,I0)') '# event = ', event_no
        do i = 1, N
            write(unit_no, '(2F12.5, I4)') x(i), y(i), atype(i)
        end do
        write(unit_no, '(A)') ''
        write(unit_no, '(A)') ''
    end subroutine write_snapshot

end program segregation
