//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Daniele Rogora
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//


/**
 * Pointer to the unique memory area allocated on the GPU to store events; it's in the constant memory
 */
__constant__ EventInfo *stack;

/**
 * A strcmp for the GPU; returns 0 if strings match
 */
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
int d_strcmp(char *s1, char *s2) {
    int ret = 0;
    while (!(ret = *(unsigned char *) s1 - *(unsigned char *) s2) && *s2) ++s1, ++s2;
    if (ret < 0)
        ret = -1;
    else if (ret > 0)
        ret = 1 ;
    return ret;
}


//This C function is needed to make the library compatible with the AC_CHECK_LIB macro
extern "C" {
  void libTRex2_with_gpu_is_present(void) {
  };
}