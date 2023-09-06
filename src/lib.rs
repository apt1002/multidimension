//! A pure-Rust library providing high-level manipulation of multi-dimensional
//! arrays.
//!
//! [`Array<I, T>`] represents an array of `T` indexed by `I`. The `T` values
//! are internally stored in a [`Box<[T]>`], which is a dense 1-dimensional
//! representation. The purpose of the `Array` wrapper is to look like a
//! multi-dimensional collection. To achieve this, the index type `I` can be
//! any type that implements [`ArrayIndex`]. This includes scalar types such as
//! `usize` and `bool`, but also tuples of other types that implement
//! `ArrayIndex`. You are encouraged to write your own index types.
//!
//! Trait [`View`] is the main way to access and manipulate [`Array`]s. Unlike
//! `Array`, a `View` doesn't store anything, but instead computes values on
//! demand. In this respect it is a bit like [`std::iter::Iterator`], and
//! indeed `View` offers some of the same methods as `Iterator`, including
//! [`View::map()`] and [`View::collect()`]. However, unlike `Iterator`,
//! `View`s are immutable; getting values using [`View::at()`] does not mutate
//! the `View`. You are encouraged to use `View`s compositionally, like
//! `Iterator`s, and to `collect()` the results into an `Array` only at the end
//! of a chain of operations.
//!
//! `ArrayIndex`es are often tuples or nested tuples. Trait `Isomorphic`
//! defines conversions between different tuple structures. For example,
//! `((A, B), C)` implements `Isomorphic<(A, (B, C))>`. Methods of `View`
//! exploit tuple isomorphism to ergonomically express which groups of
//! `ArrayIndex`es to apply operations to. The main thing you can't do with
//! tuple isomorphism is to reorder the `ArrayIndex`es; for this you need to
//! use [`View::transpose()`]. Either way, the code to manipulate the array
//! indices is all automatically generated at compile-time, and should mostly
//! be optimized away by the compiler.

mod index;
pub use index::{ArrayIndex, StaticIndex, All};

pub mod tuple;
pub use tuple::{NonTuple, Flatten, Isomorphic};

mod broadcast;
pub use broadcast::{Broadcast};

pub mod view;
pub use view::{View, FromView};

mod array;
pub use array::{Array};
