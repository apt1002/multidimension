//! Utilities for inter-converting similar tuple types.
//!
//! Conversions are implemented by first converting to a canonical form. For
//! example, the canonical form of `((A, B), (), C)` is `((((), A), B), C)`,
//! which is also the canonical form of `(A, B, C)`, `(A, (B, C))`, and so on.
//! These types can therefore all be converted into each other.
//!
//! Canonical forms implement trait [`Flat`]. Types with canonical forms
//! implement trait [`Flatten`]. The canonical form of `T` can be written
//! `<T as Flatten>: Flat`. Types which can be converted to/from `T` implement
//! trait [`Isomorphic<T>`].
//!
//! To be contained in a [`Flat`] type, a type must implement trait
//! [`NonTuple`]. If you want a type to play the tuple isomorphism game, it
//! should probably implement `NonTuple`. If you want a type to behave like a
//! tuple for the purposes of tuple isomorphism, you should consider just using
//! a tuple instead.

/// Implemented by [`Flatten`] types that have no tuple-like structure.
///
/// A type that implements `NonTuple` will automatically get implementations of
/// `Flatten` and `Isomorphic`, as will tuple types that contain it.
pub trait NonTuple: Sized {}

impl NonTuple for bool {}
impl NonTuple for char {}

impl NonTuple for i8 {}
impl NonTuple for i16 {}
impl NonTuple for i32 {}
impl NonTuple for i64 {}
impl NonTuple for i128 {}
impl NonTuple for isize {}

impl NonTuple for u8 {}
impl NonTuple for u16 {}
impl NonTuple for u32 {}
impl NonTuple for u64 {}
impl NonTuple for u128 {}
impl NonTuple for usize {}

impl<T: NonTuple> NonTuple for Option<T> {}

// ----------------------------------------------------------------------------

/// Implemented by types that are of the form `((((), A), B), ... )`.
pub trait Flat: Sized {}

impl Flat for () {}

impl<A: Flat, B: NonTuple> Flat for (A, B) {}

// ----------------------------------------------------------------------------

/// Convert a tuple-tree to/from a canonical form.
///
/// You probably shouldn't write any more implementations of this trait. If you
/// want a type to implement `Flatten`, it should probably just implement
/// [`NonTuple`], and take advantage of the blanket implementations.
pub trait Flatten<F: Flat=()>: Sized {
    type Flat: Flat;

    fn push(self, f: F) -> Self::Flat;

    fn pop(f: Self::Flat) -> (F, Self);
}

impl<F: Flat, T: NonTuple> Flatten<F> for T {
    type Flat = (F, Self);

    fn push(self, f: F) -> Self::Flat {
        (f, self)
    }

    fn pop(f: Self::Flat) -> (F, Self) {
        (f.0, f.1)
    }
}

impl<F: Flat> Flatten<F> for () {
    type Flat = F;

    fn push(self, f: F) -> Self::Flat { f }

    fn pop(f: Self::Flat) -> (F, Self) { (f, ()) }
}

impl<F: Flat, A: Flatten<F>> Flatten<F> for (A,) {
    type Flat = <A as Flatten<F>>::Flat;

    fn push(self, f: F) -> Self::Flat {
        self.0.push(f)
    }

    fn pop(f: Self::Flat) -> (F, Self) {
        let (f, a) = A::pop(f);
        (f, (a,))
    }
}

impl<F: Flat,
    A: Flatten<F>,
    B: Flatten<<A as Flatten<F>>::Flat>,
> Flatten<F> for (A, B) {
    type Flat = <B as Flatten<<A as Flatten<F>>::Flat>>::Flat;

    fn push(self, f: F) -> Self::Flat {
        self.1.push(self.0.push(f))
    }

    fn pop(f: Self::Flat) -> (F, Self) {
        let (f, b) = <B as Flatten<_>>::pop(f);
        let (f, a) = <A as Flatten<_>>::pop(f);
        (f, (a, b))
    }
}

impl<F: Flat,
    A: Flatten<F>,
    B: Flatten<<A as Flatten<F>>::Flat>,
    C: Flatten<<B as Flatten<<A as Flatten<F>>::Flat>>::Flat>,
> Flatten<F> for (A, B, C) {
    type Flat = <C as Flatten<<B as Flatten<<A as Flatten<F>>::Flat>>::Flat>>::Flat;

    fn push(self, f: F) -> Self::Flat {
        self.2.push(self.1.push(self.0.push(f)))
    }

    fn pop(f: Self::Flat) -> (F, Self) {
        let (f, c) = <C as Flatten<_>>::pop(f);
        let (f, b) = <B as Flatten<_>>::pop(f);
        let (f, a) = <A as Flatten<_>>::pop(f);
        (f, (a, b, c))
    }
}

// ----------------------------------------------------------------------------

/* This is unfortunately not possible.
 * https://github.com/rust-lang/rust/issues/108185
 *
 * Where we would like to write `T: Flattenable`, we instead write `T: Flatten`
 * and add `where` clauses to cope with the fallout.
trait Flattenable: for<F: Flat> Flatten<F> {}

impl<T: for<F: Flat> Flatten<F>> Flattenable for T {}
*/

// ----------------------------------------------------------------------------

fn to_flat<T: Flatten>(t: T) -> T::Flat {
    t.push(())
}

fn from_flat<T: Flatten>(f: T::Flat) -> T {
    let ((), t) = T::pop(f);
    t
}

/// Relates two types if they have the same canonical form.
pub trait Isomorphic<T: Flatten>: Flatten<Flat=T::Flat> {
    fn from_iso(t: T) -> Self { from_flat(to_flat(t)) }
    fn to_iso(self) -> T { from_flat(to_flat(self)) }
}

impl<T: Flatten, U: Flatten<Flat=T::Flat>> Isomorphic<T> for U {}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::fmt::{Debug};

    use super::*;

    fn assert_push_pop<F: Flat, T: Flatten<F>>(f: F, t: T, ft: T::Flat) where
        F: Debug + Copy + PartialEq,
        T: Debug + Copy + PartialEq,
        T::Flat: Debug + Copy + PartialEq,
    {
        assert_eq!(t.push(f), ft);
        assert_eq!(T::pop(ft), (f, t));
    }

    #[test]
    fn push_pop() {
        let init = ((), 7);
        assert_push_pop(init, false, (init, false));
        assert_push_pop(init, (), init);
        assert_push_pop(init, (3,), (init, 3));
        assert_push_pop(init, (3, false), ((init, 3), false));
        assert_push_pop(init, (3, false, ()), ((init, 3), false));
    }

    fn assert_flatten<T: Debug + Copy + PartialEq + Flatten>(t: T) {
        assert_eq!(t, from_flat(to_flat(t)));
    }

    #[test]
    fn flatten() {
        assert_flatten(false);
        assert_flatten(());
        assert_flatten((3,));
        assert_flatten((3, false));
        assert_flatten((3, false, ()));
    }

    fn assert_isomorphic<
        T: Debug + Copy + PartialEq + Flatten,
        U: Debug + Copy + PartialEq + Isomorphic<T>,
    >(t: T, u: U) {
        assert_eq!(t, u.to_iso());
        assert_eq!(U::from_iso(t), u);
    }

    #[test]
    fn isomorphic() {
        let a = (1, (false, ()), 2);
        let b = ((1, ()), (false, 2));
        let c = ((), (1, (false, 2)));
        let d = (((1, false), 2), ());
        assert_isomorphic(a, a);
        assert_isomorphic(a, b);
        assert_isomorphic(a, c);
        assert_isomorphic(a, d);
        assert_isomorphic(b, a);
        assert_isomorphic(b, b);
        assert_isomorphic(b, c);
        assert_isomorphic(b, d);
        assert_isomorphic(c, a);
        assert_isomorphic(c, b);
        assert_isomorphic(c, c);
        assert_isomorphic(c, d);
        assert_isomorphic(d, a);
        assert_isomorphic(d, b);
        assert_isomorphic(d, c);
        assert_isomorphic(d, d);
    }
}
