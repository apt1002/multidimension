use std::fmt::{Debug};

use super::{div_mod, Isomorphic};

/// The run-time size of an array axis. An array axis with a compile-time
/// constant size can simply use `()`, which implements this trait.
///
/// Types that implement `Size` should implement [`Isomorphic`]. The simplest
/// and best way to achieve this for a non-tuple type is to implement
/// [`NonTuple`].
///
/// [`NonTuple`]: super::NonTuple
pub trait Size: Debug + Copy + PartialEq {
    /// An abbreviation for `I::each(self, f)`.
    fn each<I: Index<Size=Self>>(self, f: impl FnMut(I)) { I::each(self, f); }
}

impl Size for usize {}
impl Size for () {}
impl<A: Size> Size for (A,) {}
impl<A: Size, B: Size> Size for (A, B) {}
impl<A: Size, B: Size, C: Size> Size for (A, B, C) {}

// ----------------------------------------------------------------------------

/// Implemented by types that can be used as an index for an [`Array`].
///
/// You are encouraged to write new `Index` types. If [`Index::Size`]
/// is a compile-time constant, you can save some effort by implementing
/// [`StaticIndex`] instead.
///
/// Types that implement `Index` should implement [`Isomorphic`]. The simplest
/// and best way to achieve this for a non-tuple type is to implement
/// [`NonTuple`].
///
/// [`Array`]: super::Array
/// [`NonTuple`]: super::NonTuple
pub trait Index: Debug + Copy + PartialEq {
    /// The run-time representation of the size of an `Array<Self, T>`.
    ///
    /// If the size is a compile-time constant, this will implement
    /// [`Isomorphic<()>`].
    type Size: Size;

    /// Returns the number of `T`s in an `Array<Self, T>`.
    fn length(size: Self::Size) -> usize;

    /// Returns the index (in `0..length()`) of `Self`.
    ///
    /// Panics if `Self` is not a valid index into an `Array` of size `size`.
    fn to_usize(self, size: Self::Size) -> usize;

    /// Returns `index / self.length(size)` and the `Self` for which
    /// `to_usize()` returns `index % self.length(size)`.
    fn from_usize(size: Self::Size, index: usize) -> (usize, Self);

    /// Equivalent to, but often more efficient than,
    /// ```text
    /// for i in 0..Self::length(size) { f(Self::from_usize(size, i).1); }
    /// ```
    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        for i in 0..Self::length(size) { f(Self::from_usize(size, i).1); }
    }

    /// Returns a View of the specified size that maps every `Self` to itself.
    fn all(size: impl Isomorphic<Self::Size>) -> All<Self> { All(size.to_iso()) }
}

// `impl Index for ()` is provided by implementing `StaticIndex`.

impl<I: Index> Index for (I,) {
    type Size = (I::Size,);

    fn length(size: Self::Size) -> usize {
        I::length(size.0)
    }

    fn to_usize(self, size: Self::Size) -> usize {
        let index = 0;
        let index = index * I::length(size.0) + self.0.to_usize(size.0);
        index
    }

    fn from_usize(size: Self::Size, index: usize) -> (usize, Self) {
        let (index, i) = I::from_usize(size.0, index);
        (index, (i,))
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        size.0.each(|i| f((i,)));
    }
}    

impl<I: Index, J: Index> Index for (I, J) {
    type Size = (I::Size, J::Size);

    fn length(size: Self::Size) -> usize {
        I::length(size.0) * J::length(size.1)
    }

    fn to_usize(self, size: Self::Size) -> usize {
        let index = 0;
        let index = index * I::length(size.0) + self.0.to_usize(size.0);
        let index = index * J::length(size.1) + self.1.to_usize(size.1);
        index
    }

    fn from_usize(size: Self::Size, index: usize) -> (usize, Self) {
        let (index, j) = J::from_usize(size.1, index);
        let (index, i) = I::from_usize(size.0, index);
        (index, (i, j))
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        size.0.each(|i| size.1.each(|j| f((i, j))));
    }
}

impl<I: Index, J: Index, K: Index> Index for (I, J, K) {
    type Size = (I::Size, J::Size, K::Size);

    fn length(size: Self::Size) -> usize {
        I::length(size.0) * J::length(size.1) * K::length(size.2)
    }

    fn to_usize(self, size: Self::Size) -> usize {
        let index = 0;
        let index = index * I::length(size.0) + self.0.to_usize(size.0);
        let index = index * J::length(size.1) + self.1.to_usize(size.1);
        let index = index * K::length(size.2) + self.2.to_usize(size.2);
        index
    }

    fn from_usize(size: Self::Size, index: usize) -> (usize, Self) {
        let (index, k) = K::from_usize(size.2, index);
        let (index, j) = J::from_usize(size.1, index);
        let (index, i) = I::from_usize(size.0, index);
        (index, (i, j, k))
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        size.0.each(|i| size.1.each(|j| size.2.each(|k| f((i, j, k)))));
    }
}

// ----------------------------------------------------------------------------

/// The return type of [`Index::all()`].
#[derive(Debug, Copy, Clone)]
pub struct All<I: Index>(I::Size);

impl<I: Index> super::View for All<I> {
    type I = I;
    type T = I;
    fn size(&self) -> <Self::I as Index>::Size { self.0 }
    fn at(&self, index: Self::I) -> Self::T { index }
}

// ----------------------------------------------------------------------------

/// Implemented by types that can be used as an index for an [`Array`] and
/// where the size of the `Array` is a compile-time constant.
///
/// [`Array`]: super::Array
pub trait StaticIndex: 'static + Index<Size=()> {
    /// The (compile-time constant) complete list of `Self`s.
    const ALL: &'static [Self];

    /// Returns the index in `ALL` of `Self`.
    fn to_usize(self) -> usize;

    /// Equivalent to, but often more efficient than, `ALL[index]`.
    fn from_usize(index: usize) -> Self { Self::ALL[index] }
}

impl<I: StaticIndex> Index for I {
    type Size = ();

    #[inline(always)] // Want the caller to see this as constant.
    fn length((): Self::Size) -> usize { Self::ALL.len() }

    fn to_usize(self, (): Self::Size) -> usize {
        let index = StaticIndex::to_usize(self);
        assert_eq!(self, Self::ALL[index]);
        index
    }

    #[inline(always)] // Want the caller to optimise if `index < length()`.
    fn from_usize(_: Self::Size, index: usize) -> (usize, Self) {
        let (q, r) = div_mod(index, Self::length(()));
        (q, StaticIndex::from_usize(r))
    }

    // Compiler might work this out by itself, but why leave it uncertain?
    fn each(_: Self::Size, mut f: impl FnMut(Self)) {
	for i in 0..Self::length(()) { f(StaticIndex::from_usize(i)); }
    }
}

impl StaticIndex for () {
    const ALL: &'static [Self] = &[()];
    fn to_usize(self) -> usize { 0 }
    fn from_usize(_: usize) -> Self { () }
}

impl StaticIndex for bool {
    const ALL: &'static [Self] = &[false, true];
    fn to_usize(self) -> usize { self as usize }
    fn from_usize(index: usize) -> Self { index != 0 }
}
