//! A variety of [`Index`]es and [`View`]s based on `usize`.
//!
//! `usize` itself implements `Index`.

use super::{NonTuple, Index, View};

impl Index for usize {
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

/// An `I`-like [`Index`] with a compile-time constant size.
///
/// This wrapper can be applied to any `Index` whose `Size` is a `usize`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fixed<I, const SIZE: usize>(pub I);

impl<I, const SIZE: usize> NonTuple for Fixed<I, SIZE> {}

impl<I: Index<Size=usize>, const SIZE: usize> Index for Fixed<I, SIZE> {
    type Size = ();
    fn length(_: ()) -> usize { I::length(SIZE) }
    fn as_usize(self, _: ()) -> usize { self.0.as_usize(SIZE) }
    fn each(_: (), mut f: impl FnMut(Self)) { I::each(SIZE, |i| f(Fixed(i))) }
}

// ----------------------------------------------------------------------------

/// A `usize`-like [`Index`] that counts backwards.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reversed(pub usize);

impl NonTuple for Reversed {}

impl Index for Reversed {
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
/// The inverse map is `<Self::T as Index>::as_usize`.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct DivMod<const B: usize>(usize);

impl<const B: usize> View for DivMod<B> {
    type I = usize;
    type T = (usize, Fixed<usize, B>);
    fn size(&self) -> <Self::I as Index>::Size { self.0 * B }
    fn at(&self, index: Self::I) -> Self::T { (index / B, Fixed(index % B)) }
}
