//! A variety of [`Index`]es and [`View`]s based on `usize`.
//!
//! `usize` itself implements `Index`.

use super::{div_mod, NonTuple, Size, Index, View};

impl Index for usize {
    type Size = usize;

    fn length(size: Self::Size) -> usize { size }

    fn to_usize(self, size: Self::Size) -> usize {
        assert!(self < size, "Index {:?} is out of bounds for size {:?}", self, size);
        self
    }

    fn from_usize(size: Self::Size, index: usize) -> (usize, Self) { div_mod(index, size) }

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

    fn to_usize(self, _: ()) -> usize { self.0.to_usize(SIZE) }

    fn from_usize(_: Self::Size, index: usize) -> (usize, Self) {
        let (index, i) = I::from_usize(SIZE, index);
        (index, Fixed(i))
    }

    fn each(_: (), mut f: impl FnMut(Self)) { SIZE.each(|i| f(Fixed(i))) }
}

// ----------------------------------------------------------------------------

fn reverse(size: usize, index: usize) -> usize {
    (size - 1) - index
}

/// A `usize`-like [`Index`] that counts backwards.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reversed(pub usize);

impl NonTuple for Reversed {}

impl Index for Reversed {
    type Size = usize;

    fn length(size: Self::Size) -> usize { size }

    fn to_usize(self, size: Self::Size) -> usize { reverse(size, self.0) }

    fn from_usize(size: Self::Size, index: usize) -> (usize, Self) {
        let (q, r) = div_mod(index, size);
        (q, Reversed(reverse(size, r)))
    }

    fn each(size: Self::Size, mut f: impl FnMut(Self)) {
        for i in 0..size { f(Reversed(reverse(size, i))); }
    }
}
