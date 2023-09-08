//! A variety of [`ArrayIndex`]es and [`View`]s based on `usize`.
//!
//! `usize` itself implements `ArrayIndex`.

use super::{NonTuple, ArrayIndex, View};

impl ArrayIndex for usize {
    type Size = usize;

    fn length(size: Self::Size) -> usize { size }

    fn as_usize(self, size: Self::Size) -> usize {
        assert!(self < size, "Index {:?} is out of bounds for size {:?}", self, size);
        self
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        for i in 0..size { f(i); }
    }
}

// ----------------------------------------------------------------------------

/// An `I`-like [`ArrayIndex`] with a compile-time constant size.
///
/// This wrapper can be applied to any `ArrayIndex` whose `Size` is a `usize`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed<I, const SIZE: usize>(pub I);

impl<I, const SIZE: usize> NonTuple for Fixed<I, SIZE> {}

impl<I: ArrayIndex<Size=usize>, const SIZE: usize> ArrayIndex for Fixed<I, SIZE> {
    type Size = ();
    fn length(_: ()) -> usize { I::length(SIZE) }
    fn as_usize(self, _: ()) -> usize { self.0.as_usize(SIZE) }
    fn each(_: (), mut f: impl FnMut(Self)) { I::each(SIZE, |i| f(Fixed(i))) }
}

// ----------------------------------------------------------------------------

/// A `usize`-like [`ArrayIndex`] that counts backwards.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reversed(pub usize);

impl NonTuple for Reversed {}

impl ArrayIndex for Reversed {
    type Size = usize;

    fn length(size: Self::Size) -> usize { size }

    fn as_usize(self, size: Self::Size) -> usize {
        assert!(self.0 < size, "Index {:?} is out of bounds for size {:?}", self, size);
        (size - 1) - self.0
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        for i in 0..size { f(Reversed((size - 1) - i)); }
    }
}

// ----------------------------------------------------------------------------

/// A [`View`] that maps each `index` to `(index / B, Fixed(index % B))`.
///
/// The inverse map is `<Self::T as ArrayIndex>::as_usize`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct DivMod<const B: usize>(usize);

impl<const B: usize> View for DivMod<B> {
    type I = usize;
    type T = (usize, Fixed<usize, B>);
    fn size(&self) -> <Self::I as ArrayIndex>::Size { self.0 * B }
    fn at(&self, index: Self::I) -> Self::T { (index / B, Fixed(index % B)) }
}
