use std::fmt::{Debug};

use super::{Flatten};

/// Implemented by types that can be used as an index for an [`Array`].
///
/// You are encouraged to write new `ArrayIndex` types. If [`ArrayIndex::Size`]
/// is a compile-time constant, you can save some effort by implementing
/// [`StaticIndex`] instead.
///
/// All types that implement `ArrayIndex` must implement [`Flatten`]. The
/// simplest way to achieve this for a non-tuple type is to implement
/// [`NonTuple`].
///
/// [`Array`]: super::Array
/// [`NonTuple`]: super::NonTuple
pub trait ArrayIndex: Copy + Flatten {
    /// The run-time representation of the size of an `Array<Self, T>`.
    ///
    /// If the size is a compile-time constant, this will implement
    /// [`Isomorphic<()>`].
    type Size: Copy + PartialEq + Flatten;

    /// Returns the number of `T`s in an `Array<Self, T>`.
    fn length(size: Self::Size) -> usize;

    /// Returns the index (in `0..length()`) of `Self`.
    ///
    /// Panics if `Self` is not a valid index into an `Array` of size `size`.
    fn as_usize(self, size: Self::Size) -> usize;

    /// For `i` in `0..Self::length(size)`, apply `f` to the `Self` for which
    /// `as_usize()` returns `i`.
    fn each(size: Self::Size, f: impl FnMut(Self));

    /// Returns a View of the specified size that maps every `Self` to itself.
    fn all(size: Self::Size) -> All<Self> { All(size) }
}

impl ArrayIndex for usize {
    type Size = usize;

    fn length(size: Self::Size) -> usize { size }

    fn as_usize(self, size: Self::Size) -> usize {
        assert!(self < size, "Index {} is out of bounds for size {}", self, size);
        self
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        for i in 0..size { f(i); }
    }
}

impl<I: ArrayIndex> ArrayIndex for (I,) where
    (I,): Flatten,
    (I::Size,): Flatten,
{
    type Size = (I::Size,);

    fn length(size: Self::Size) -> usize {
        I::length(size.0)
    }

    fn as_usize(self, size: Self::Size) -> usize {
        let index = self.0.as_usize(size.0);
        index
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        I::each(size.0, |i| f((i,)));
    }
}
    

impl<I: ArrayIndex, J: ArrayIndex> ArrayIndex for (I, J) where
    (I, J): Flatten,
    (I::Size, J::Size): Flatten,
{
    type Size = (I::Size, J::Size);

    fn length(size: Self::Size) -> usize {
        I::length(size.0) * J::length(size.1)
    }

    fn as_usize(self, size: Self::Size) -> usize {
        let index = self.0.as_usize(size.0);
        let index = index * J::length(size.1) + self.1.as_usize(size.1);
        index
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        I::each(size.0, |i| J::each(size.1, |j| f((i, j))));
    }
}

impl<I: ArrayIndex, J: ArrayIndex, K: ArrayIndex> ArrayIndex for (I, J, K) where
    (I, J, K): Flatten,
    (I::Size, J::Size, K::Size): Flatten,
{
    type Size = (I::Size, J::Size, K::Size);

    fn length(size: Self::Size) -> usize {
        I::length(size.0) * J::length(size.1) * K::length(size.2)
    }

    fn as_usize(self, size: Self::Size) -> usize {
        let index = self.0.as_usize(size.0);
        let index = index * J::length(size.1) + self.1.as_usize(size.1);
        let index = index * K::length(size.2) + self.2.as_usize(size.2);
        index
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        I::each(size.0, |i| J::each(size.1, |j| K::each(size.2, |k| f((i, j, k)))));
    }
}

// ----------------------------------------------------------------------------

/// The return type of [`ArrayIndex::all()`].
#[derive(Debug, Copy, Clone)]
pub struct All<I: ArrayIndex>(I::Size);

impl<I: ArrayIndex> super::View for super::index::All<I> {
    type I = I;
    type T = I;
    fn size(&self) -> <Self::I as ArrayIndex>::Size { self.0 }
    fn at(&self, index: Self::I) -> Self::T { index }
}

// ----------------------------------------------------------------------------

/// Implemented by types that can be used as an index for an [`Array`] and
/// where the size of the `Array` is a compile-time constant.
///
/// [`Array`]: super::Array
pub trait StaticIndex: 'static + Debug + Copy + PartialEq + Flatten {
    /// The (compile-time constant) complete list of `Self`s.
    const ALL: &'static [Self];

    /// Returns the index in `ALL` of `Self`.
    fn as_usize(self) -> usize;
}

impl<I: StaticIndex> ArrayIndex for I {
    type Size = ();

    fn length((): Self::Size) -> usize { Self::ALL.len() }

    fn as_usize(self, (): Self::Size) -> usize {
        let index = StaticIndex::as_usize(self);
        assert_eq!(self, Self::ALL[index]);
        index
    }

    fn each(_: Self::Size, mut f: impl FnMut(Self)) {
        for &i in Self::ALL { f(i); }
    }
}

impl StaticIndex for () {
    const ALL: &'static [Self] = &[()];

    fn as_usize(self) -> usize { 0 }
}

impl StaticIndex for bool {
    const ALL: &'static [Self] = &[false, true];

    fn as_usize(self) -> usize { self as usize }
}
