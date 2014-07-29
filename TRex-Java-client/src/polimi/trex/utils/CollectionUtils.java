//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Francesco Feltrinelli
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

package polimi.trex.utils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Various helper methods for collections and primitive arrays.
 */
public class CollectionUtils {

	/**
	 * Copy the given array to a dynamic {@link List}.
	 */
	public static <T> List<T> copyToList(T[] array){
		List<T> list= new ArrayList<T>(array.length);
		for (T elem: array){
			list.add(elem);
		}
		return list;
	}
	
	public static byte[] concat(byte single, byte[] array) {
		byte[] result= new byte[1 + array.length];
		result[0]= single;
		System.arraycopy(array, 0, result, 1, array.length);
		return result;
	}
	
	public static byte[] concat(byte[] array, byte single) {
		byte[] result = new byte[1 + array.length];
		System.arraycopy(array, 0, result, 0, array.length);
		result[array.length]= single;
		return result;
	}
	
	public static byte[] concat(byte[] first, byte[] second) {
		byte[] result = new byte[first.length + second.length];
		System.arraycopy(first, 0, result, 0, first.length);
		System.arraycopy(second, 0, result, first.length, second.length);
		return result;
	}
	
	public static byte[] concat(
			byte[] first, int firstStart, int firstLength,
			byte[] second, int secondStart, int secondLength) {
		byte[] result = new byte[firstLength + secondLength];
		System.arraycopy(first, firstStart, result, 0, firstLength);
		System.arraycopy(second, secondStart, result, firstLength, secondLength);
		return result;
	}
	
	public static byte[] concat(byte[] first, byte[] second, int secondStart, int secondLength){
		return concat(first, 0, first.length, second, secondStart, secondLength);
	}
	
	public static byte[] concat(byte[] first, int firstStart, int firstLength, byte[] second) {
		return concat(first, firstStart, firstLength, second, 0, second.length);
	}
	
	public static <T> T[] concat(T single, T[] array) {
		@SuppressWarnings("unchecked")
		final T[] result = (T[]) Array.newInstance(single.getClass()
				.getComponentType(), 1 + array.length);

		result[0]= single;
		System.arraycopy(array, 0, result, 1, array.length);
		
		return result;
	}
	
	public static <T> T[] concat(T[] array, T single) {
		@SuppressWarnings("unchecked")
		final T[] result = (T[]) Array.newInstance(single.getClass()
				.getComponentType(), 1 + array.length);

		System.arraycopy(array, 0, result, 0, array.length);
		result[array.length]= single;
		
		return result;
	}
	
	public static <T> T[] concat(T[] first, T[] second) {
		@SuppressWarnings("unchecked")
		final T[] result = (T[]) Array.newInstance(first.getClass()
				.getComponentType(), first.length + second.length);

		System.arraycopy(first, 0, result, 0, first.length);
		System.arraycopy(second, 0, result, first.length, second.length);
		
		return result;
	}

	public static <T> T[] concatAll(T[] first, T[]... rest) {
		int totalLength = first.length;
		for (T[] array : rest) {
			totalLength += array.length;
		}

		@SuppressWarnings("unchecked")
		final T[] result = (T[]) Array.newInstance(first.getClass()
				.getComponentType(), totalLength);

		System.arraycopy(first, 0, result, 0, first.length);
		int offset = first.length;
		for (T[] array : rest) {
			System.arraycopy(array, 0, result, offset, array.length);
			offset += array.length;
		}

		return result;
	}
	
	public static byte[] subset(byte[] array, int start, int length) {
		byte[] result = new byte[length];
		System.arraycopy(array, start, result, 0, length);
		return result;
	}
	
	@SuppressWarnings("unchecked")
	public static <S,D extends S> List<D> downcast(Collection<S> source, Class<D> destClass){
		List<D> dest= new ArrayList<D>();
		for (S s: source){
			if (destClass.isAssignableFrom(s.getClass())) dest.add((D) s);
			else throw new IllegalArgumentException("Could not downcast element from class "+s.getClass().getSimpleName()+" to "+destClass.getSimpleName());
		}
		return dest;
	}
	
	@SuppressWarnings("unchecked")
	public static <S,D extends S> List<D> downcast(S[] source, Class<D> destClass){
		List<D> dest= new ArrayList<D>();
		for (S s: source){
			if (destClass.isAssignableFrom(s.getClass())) dest.add((D) s);
			else throw new IllegalArgumentException("Could not downcast element from class "+s.getClass().getSimpleName()+" to "+destClass.getSimpleName());
		}
		return dest;
	}
	
	@SuppressWarnings("unchecked")
	public static <S,D extends S> D[] downcastToArray(Collection<S> source, Class<D> destClass){
		D[] dest = (D[]) Array.newInstance(destClass, source.size());
		Iterator<S> it= source.iterator();
		for (int i=0; i<dest.length; i++){
			S s= it.next();
			if (destClass.isAssignableFrom(s.getClass())) dest[i]= (D) s;
			else throw new IllegalArgumentException("Could not downcast element from class "+s.getClass().getSimpleName()+" to "+destClass.getSimpleName());
		}
		return dest;
	}
	
	public static <D,S extends D> List<D> upcast(Collection<S> source, Class<D> destClass){
		List<D> dest= new ArrayList<D>();
		for (S s: source){
			dest.add(s);
		}
		return dest;
	}
	
	public static <D,S extends D> List<D> upcast(S[] source, Class<D> destClass){
		List<D> dest= new ArrayList<D>();
		for (S s: source){
			dest.add(s);
		}
		return dest;
	}
}
