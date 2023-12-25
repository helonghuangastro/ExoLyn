module matrixsol

implicit none
contains


subroutine sol(X, B)
    ! implicit none
    !integer, intent(in) :: nvar, N
    ! double precision, intent(inout) :: X(N, nvar, 9)
    ! double precision, intent(inout) :: B(nvar, nvar)
    double precision, intent(inout) :: X(:,:,:)
    double precision, intent(inout) :: B(:,:)
    integer :: i,j,k,N, nvar
    !real(8), intent(out) :: sol(:,:)

    N = size(X,1)
    nvar = size(X,2)

    ! start to manipulate the inverse matrix
    do i=1, N
        ! eliminate until p2
        do j=1, nvar
            B(j, i) = B(j, i)/X(i, j, j+nvar)
            X(i, j, (j+nvar):) = X(i, j, (j+nvar):)/X(i, j, j+nvar)
            do k=(j+1), nvar
                B(k, i) = B(k, i) - X(i, k, j+nvar) * B(j, i)
                X(i, k, (j+nvar):) = X(i, k, (j+nvar):) - X(i, k, (j+nvar)) * X(i, j, (j+nvar):)
            end do
        end do

        ! eliminate until p3
        do j=2, nvar
            do k=1, j-1
                B(k, i)  = B(k, i) - B(j, i) * X(i, k, j+nvar)
                X(i, k, (j+nvar):) = X(i, k, (j+nvar):) - X(i, k, (j+nvar)) * X(i, j, (j+nvar):)
            end do
        end do

        ! stop before the last block
        if(i==N) exit

        ! eliminate until p4
        do j=1, nvar
            do k=1, nvar
                B(k, i+1) = B(k, i+1) - B(j, i) * X(i+1, k, j)
                X(i+1, k, :(2*nvar)) = X(i+1, k, :(2*nvar)) - X(i+1, k, j) * X(i, j, (nvar+1):)
            end do
        end do
    end do

    X(N, :, (2*nvar+1):) = 0
    ! eliminate everything
    do i=N,2,-1
        do j=1,nvar
            B(:, i-1) = B(:, i-1) - X(i-1, :, (j+2*nvar)) * B(j, i)
            X(i-1, :, (j+2*nvar)) = 0
        end do
    end do

    ! sol = B

end subroutine sol
! end subroutine sol

end module