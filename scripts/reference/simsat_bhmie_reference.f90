program simsat_bhmie_reference
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer :: id, nang, ios, j
  real :: x, nr, ni, qext, qsca, qback, g
  complex :: m
  complex, allocatable :: s1(:), s2(:)
  real(kind=8) :: pi, theta, phase
  allocate(s1(199999), s2(199999))
  pi = 4d0*atan(1d0)
  write(*,'(A)') 'id,x,n_real,n_imag,angle_deg,qext,qsca,qback_differential,g,phase_sr1'
  do
    read(*,*,iostat=ios) id,x,nr,ni,nang
    if (ios < 0) exit
    if (ios /= 0) stop 2
    if (.not.ieee_is_finite(x) .or. x<=0 .or. x>10000) stop 3
    if (.not.ieee_is_finite(nr) .or. nr<=0 .or. .not.ieee_is_finite(ni) .or. ni<0) stop 3
    if (nang<2 .or. nang>1000) stop 3
    m=cmplx(nr,ni)
    call bhmie(x,m,nang,s1,s2,qext,qsca,qback,g)
    do j=1,2*nang-1
      theta=90d0*dble(j-1)/dble(nang-1)
      phase=(dble(real(s1(j)))**2+dble(aimag(s1(j)))**2 + &
             dble(real(s2(j)))**2+dble(aimag(s2(j)))**2)/(2d0*pi*dble(x)**2*dble(qsca))
      if (.not.ieee_is_finite(phase)) stop 4
      write(*,'(I0,9(",",ES25.17E3))') id,dble(x),dble(nr),dble(ni),theta,dble(qext),dble(qsca),dble(qback),dble(g),phase
    end do
  end do
end program
