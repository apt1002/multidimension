//! Generic programming over binary arithmetic operators.
//!
//! For each binary operator in [`std::ops`] this module contains a type of the
//! same name that cannot be instantiated and that implements [`Binary`]. For
//! example, [`Add`] corresponds to [`std::ops::Add`]. This can be passed as a
//! type parameter to generic code, e.g. [`View::binary()`].
//!
//! [`View::binary()`]: super::View::binary()

/// A function that combines `T` with `U`.
///
/// This trait has no methods that take `self`. It makes sense to implement it
/// for types that cannot be instantiated, such as empty enumerations.
pub trait Binary<T, U> {
    type Output;

    fn call(t: T, u: U) -> Self::Output;
}

// ----------------------------------------------------------------------------

/// An implementation of [`Binary`] that constructs a pair.
pub enum Pair {}

impl<T, U> Binary<T, U> for Pair {
    type Output = (T, U);
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { (t, u) }
}

// ----------------------------------------------------------------------------

pub enum Add {}

impl<T, U> Binary<T, U> for Add where T: std::ops::Add<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.add(u) }
}

// ----------------------------------------------------------------------------

pub enum Sub {}

impl<T, U> Binary<T, U> for Sub where T: std::ops::Sub<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.sub(u) }
}

// ----------------------------------------------------------------------------

pub enum Mul {}

impl<T, U> Binary<T, U> for Mul where T: std::ops::Mul<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.mul(u) }
}

// ----------------------------------------------------------------------------

pub enum Div {}

impl<T, U> Binary<T, U> for Div where T: std::ops::Div<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.div(u) }
}

// ----------------------------------------------------------------------------

pub enum Rem {}

impl<T, U> Binary<T, U> for Rem where T: std::ops::Rem<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.rem(u) }
}

// ----------------------------------------------------------------------------

pub enum BitAnd {}

impl<T, U> Binary<T, U> for BitAnd where T: std::ops::BitAnd<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.bitand(u) }
}

// ----------------------------------------------------------------------------

pub enum BitOr {}

impl<T, U> Binary<T, U> for BitOr where T: std::ops::BitOr<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.bitor(u) }
}

// ----------------------------------------------------------------------------

pub enum BitXor {}

impl<T, U> Binary<T, U> for BitXor where T: std::ops::BitXor<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.bitxor(u) }
}

// ----------------------------------------------------------------------------

pub enum Shl {}

impl<T, U> Binary<T, U> for Shl where T: std::ops::Shl<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.shl(u) }
}

// ----------------------------------------------------------------------------

pub enum Shr {}

impl<T, U> Binary<T, U> for Shr where T: std::ops::Shr<U> {
    type Output = T::Output;
    #[inline(always)]
    fn call(t: T, u: U) -> Self::Output { t.shr(u) }
}

// ----------------------------------------------------------------------------

/// Implement one of the [`std::ops`] traits for a type that implements
/// [`View`].
///
/// You perhaps want to use [`impl_ops_for_view`] instead, which calls this.
///
/// [`View`]: super::View
///
/// ```
/// use multidimension::{Index, View, impl_op_for_view};
///
/// pub struct VecView<T: Clone>(Vec<T>);
///
/// impl<T: Clone> View for VecView<T> {
///     type I = usize;
///     type T = T;
///     fn size(&self) -> usize { self.0.len() }
///     fn at(&self, index: usize) -> T { self.0[index].clone() }
/// }
///
/// impl_op_for_view! { Add for VecView<T: Clone> { add } }
/// impl_op_for_view! { BitXor for VecView<T: Clone> { bitxor } }
/// // Etc.
/// ```
///
/// [`impl_ops_for_view`]: crate::impl_ops_for_view
#[macro_export]
macro_rules! impl_op_for_view {
    ($op:ident for $v:ident<$($a:lifetime,)? $($param:ident$(: $bound:path)?),*> { $method:ident }) => {
        impl<$($a,)? RHS: $crate::View, $($param$(: $bound)?),*> std::ops::$op<RHS> for $v<$($a,)? $($param),*> where
            Self: $crate::View,
            <Self as View>::I: $crate::Broadcast<RHS::I>,
            <Self as View>::T: std::ops::$op<RHS::T>,
        {
            type Output = $crate::view::Zip<Self, RHS, $crate::ops::$op>;
            fn $method(self, other: RHS) -> Self::Output { self.binary(other) }
        }
    };
}

/// Implement all of the [`std::ops`] traits for a type that implements
/// [`View`]. The implementations call [`View::binary()`].
///
/// The macro has some syntactic limitations, but should usually work.
///
/// ```
/// use multidimension::{Index, View, impl_ops_for_view};
///
/// pub struct VecView<T: Clone>(Vec<T>);
///
/// impl<T: Clone> View for VecView<T> {
///     type I = usize;
///     type T = T;
///     fn size(&self) -> usize { self.0.len() }
///     fn at(&self, index: usize) -> T { self.0[index].clone() }
/// }
///
/// impl_ops_for_view!(VecView<T: Clone>);
/// ```
///
/// [`View`]: super::View
/// [`View::binary()`]: super::View::binary()
#[macro_export]
macro_rules! impl_ops_for_view {
    ($v:ident<$($a:lifetime,)? $($param:ident$(: $bound:path)?),*>) => {
        $crate::impl_op_for_view! { Add for $v<$($a,)? $($param$(: $bound)?),*> { add } }
        $crate::impl_op_for_view! { Sub for $v<$($a,)? $($param$(: $bound)?),*> { sub } }
        $crate::impl_op_for_view! { Mul for $v<$($a,)? $($param$(: $bound)?),*> { mul } }
        $crate::impl_op_for_view! { Div for $v<$($a,)? $($param$(: $bound)?),*> { div } }
        $crate::impl_op_for_view! { Rem for $v<$($a,)? $($param$(: $bound)?),*> { rem } }
        $crate::impl_op_for_view! { BitAnd for $v<$($a,)? $($param$(: $bound)?),*> { bitand } }
        $crate::impl_op_for_view! { BitOr for $v<$($a,)? $($param$(: $bound)?),*> { bitor } }
        $crate::impl_op_for_view! { BitXor for $v<$($a,)? $($param$(: $bound)?),*> { bitxor } }
        $crate::impl_op_for_view! { Shl for $v<$($a,)? $($param$(: $bound)?),*> { shl } }
        $crate::impl_op_for_view! { Shr for $v<$($a,)? $($param$(: $bound)?),*> { shr } }
    };
}

// ----------------------------------------------------------------------------

/// A helper macro for implementing [`std::ops::Index`] and
/// [`std::ops::IndexMut`] in terms of [`View`] and [`ViewMut`].
///
/// ```
/// use multidimension::{Index, View, ViewRef, ViewMut, impl_ops_for_memoryview};
///
/// pub struct VecView<T: Clone>(Vec<T>);
///
/// impl<T: Clone> View for VecView<T> {
///     type I = usize;
///     type T = T;
///     fn size(&self) -> usize { self.0.len() }
///     fn at(&self, index: usize) -> T { self.0[index].clone() }
/// }
///
/// impl<T: Clone> ViewRef for VecView<T> {
///     fn at_ref(&self, index: Self::I) -> &Self::T { &self.0[index] }
/// }
///
/// impl<T: Clone> ViewMut for VecView<T> {
///     fn at_mut(&mut self, index: Self::I) -> &mut Self::T { &mut self.0[index] }
/// }
///
/// impl_ops_for_memoryview!(VecView<T: Clone>);
/// ```
#[macro_export]
macro_rules! impl_ops_for_memoryview {
    ($v:ident<$($a:lifetime,)? $($param:ident$(: $bound:path)?),*>) => {
        impl<$($a,)? $($param$(: $bound)?),*> std::ops::Index<<Self as View>::I> for $v<$($a,)? $($param),*> where
            Self: $crate::ViewRef,
        {
            type Output = <Self as View>::T;
            fn index(&self, index: <Self as View>::I) -> &Self::Output { self.at_ref(index) }
        }

        impl<$($a,)? $($param$(: $bound)?),*> std::ops::IndexMut<<Self as View>::I> for $v<$($a,)? $($param),*> where
            Self: $crate::ViewMut,
        {
            fn index_mut(&mut self, index: <Self as View>::I) -> &mut Self::Output { self.at_mut(index) }
        }
    };
}
