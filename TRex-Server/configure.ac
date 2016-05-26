AC_INIT([TRexServer], [1.0], [daniele.rogora@gmail.com])
AM_INIT_AUTOMAKE([subdir-objects foreign -Wall])
CXXFLAGS="-O3"
AX_CXX_COMPILE_STDCXX(11, noext, mandatory)
AM_PROG_AR
AC_PROG_CC
AC_PROG_CXX
AC_CONFIG_MACRO_DIR([m4])
AC_PROG_LIBTOOL
AC_LANG_PUSH([C++])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile])

AC_ARG_WITH(
    TRex2,
    AS_HELP_STRING(
	[--with-TRex2=DIR],
	[TRex2 source directory]
    ),
    [
	TRex2="$withval"
    ]
)
CFLAGS="$CFLAGS -I${TRex2}/src -I/usr/include/TRex2 -I/usr/local/include/TRex2"
CXXFLAGS="$CXXFLAGS -I${TRex2}/src -I${TRex2}/include/TRex2 -I/usr/include/TRex2 -I/usr/local/include/TRex2"
CPPFLAGS="$CXXFLAGS -I${TRex2}/src -I${TRex2}/include/TRex2 -I/usr/include/TRex2 -I/usr/local/include/TRex2"
LDFLAGS="$LDFLAGS -L/usr/lib -L/usr/local/lib -L${TRex2}/src/.libs -L${TRex2}/lib"
AC_SUBST(TRexServer_CPPFLAGS)
AC_CHECK_HEADERS([Engine/TRexEngine.h], [], AC_MSG_ERROR([TRex headers not found (use --with-TRex2 option)]))
AC_CHECK_LIB([TRex2], [libTRex2_is_present], [], AC_MSG_ERROR([TRex library not found]))
BOOST_REQUIRE([1.54])
BOOST_THREAD
BOOST_ARRAY
BOOST_SYSTEM
BOOST_LOG

AC_CHECK_LIB([TRex2], [libTRex2_with_gpu_is_present], [
echo "#### GPU engine found; you can use it with the -gpu command line option ####"
AC_DEFINE([HAVE_GTREX], [], [TRex2 has enabled the GPU engine])
], [])

AC_OUTPUT 
echo "Project configured; type make to build TRexServer"
