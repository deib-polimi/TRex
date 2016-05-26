AC_INIT([TRex2], [1.0], [daniele.rogora@gmail.com])
AM_INIT_AUTOMAKE([subdir-objects foreign -Wall])
CXXFLAGS="-O3"
AX_CXX_COMPILE_STDCXX(11, noext, mandatory)
AM_PROG_AR
AC_PROG_CC
AC_PROG_CXX
AC_ENABLE_SHARED
AC_DISABLE_STATIC
AC_CONFIG_MACRO_DIR([m4])
AC_PROG_LIBTOOL
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_CHECK_LIB([pthread], [main])

AC_ARG_ENABLE([cuda], 
	      AC_HELP_STRING([--enable-cuda=ARCH],
	      [Enable Nvidia CUDA powered engine for architecture ARCH (see nvcc option -arch and README file).]),[

	      using_cuda=$enableval

	      NVCCFLAGS="-Xcompiler -fPIC -O3 -m64 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES"

	      # Search for compiler
	      AC_CHECK_PROG([cuda_compiler], [nvcc], yes, no) 
	      AS_IF([test "x$cuda_compiler" = "xno"],[
                        AC_MSG_ERROR([NVCC compiler not found! Aborting compilation.])
              ])
	      AC_PATH_PROG(NVCC,nvcc, "no")   
	      # Search for compiler
	      AC_CHECK_PROG([cuda_compiler], [nvcc], yes, no) 
	      AS_IF([test "x$cuda_compiler" = "xno"],[
	      AC_MSG_ERROR([NVCC compiler not found! Aborting compilation.])
	      ])


	      AC_CHECK_LIB([cudart], [cudaMalloc], , AC_MSG_WARN([CUDA not found the GPU engine won't be compiled]))

	      # If $enable_cuda is not empty, set to CUDA_ARCH to
              # supplied value, else set to value sm_11

	      AS_IF([test "x$enableval" = "xyes"],[
                        CUDA_ARCH=" -arch=sm_21"
			echo "compiling with nvcc for architecture sm_21 (default)"
           		],[
                	CUDA_ARCH=" -arch=$enableval"
			echo "compiling with nvcc for architecture $enableval"
			using_cuda="yes"
              ])
              NVCCFLAGS+=" $CUDA_ARCH"

              ###############################################################################
	      # Setup CUDA paths
	      # 
	      AC_ARG_WITH([cuda],
		      [--with-cuda=PATH prefix where cuda is installed [default=/usr/local/cuda]])

	      if test -n "$with_cuda"
	      then
		      CUDA_CFLAGS="-I$with_cuda/include"
		      CUDA_LIBS="-L$with_cuda/lib64 -lcudart -lcublas"
		      CUFFT_LIBS="-L$with_cuda/lib64 -lcufft"
		      CUBLAS_LIBS="-L$with_cuda/lib64 -lcublas"
		      NVCC="$with_cuda/bin/nvcc"
	      else
		      CUDA_CFLAGS="-I/usr/local/cuda/include"
		      CUDA_LIBS="-L/usr/local/cuda/lib64 -lcudart -lcublas"
		      CUFFT_LIBS="-L/usr/local/cuda/lib64 -lcufft"
		      CUBLAS_LIBS="-L/usr/local/cuda/lib64 -lcublas"
		      NVCC="nvcc"
	      fi

	      AC_SUBST(CUDA_CFLAGS)
	      AC_SUBST(CUDA_LIBS)
	      AC_SUBST(CUFFT_LIBS)
	      AC_SUBST(CUBLAS_LIBS)
	      AC_SUBST(NVCC)
	      AC_SUBST(NVCCFLAGS)

	      ################################################################################
              ])

AM_CONDITIONAL([USE_GPU], [test "$using_cuda" = yes])    
AC_SUBST(USE_GPU)
AC_OUTPUT
