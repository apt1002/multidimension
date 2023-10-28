use super::{NonTuple};

/// `Coated<I>` behaves like `I` in most respects, but implements [`NonTuple`].
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct Coated<I>(pub I);

impl<I> NonTuple for Coated<I>{}

impl<I> std::ops::Deref for Coated<I> {
    type Target = I;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<I> std::ops::DerefMut for Coated<I> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

// ----------------------------------------------------------------------------

/// Implemented by types that differ from `T` by adding or removing at most one
/// level of [`Coated`].
pub trait Coat<I: Coat<Self>>: Sized {
    fn coat(self) -> I;
}

impl<I> Coat<I> for Coated<I> {
    #[inline(always)]
    fn coat(self) -> I { self.0 }
}

impl<I> Coat<Coated<I>> for I {
    #[inline(always)]
    fn coat(self) -> Coated<I> { Coated(self) }
}

impl<I: NonTuple> Coat<Self> for I {
    #[inline(always)]
    fn coat(self) -> Self { self }
}

impl Coat<()> for () {
    #[inline(always)]
    fn coat(self) -> () { () }
}

impl<
    I: Coat<CI>, CI: Coat<I>,
> Coat<(CI,)> for (I,) {
    #[inline(always)]
    fn coat(self) -> (CI,) { (self.0.coat(),) }
}

impl<
    I: Coat<CI>, CI: Coat<I>,
    J: Coat<CJ>, CJ: Coat<J>,
> Coat<(CI, CJ)> for (I, J) {
    #[inline(always)]
    fn coat(self) -> (CI, CJ) { (self.0.coat(), self.1.coat()) }
}

impl<
    I: Coat<CI>, CI: Coat<I>,
    J: Coat<CJ>, CJ: Coat<J>,
    K: Coat<CK>, CK: Coat<K>,
> Coat<(CI, CJ, CK)> for (I, J, K) {
    #[inline(always)]
    fn coat(self) -> (CI, CJ, CK) { (self.0.coat(), self.1.coat(), self.2.coat()) }
}
