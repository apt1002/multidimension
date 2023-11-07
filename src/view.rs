use std::fmt::{Debug};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};

use super::{Isomorphic, Coat as _, Index, Broadcast, Binary, impl_ops_for_view, impl_ops_for_memoryview};

/// A buffer that accumulates items of type `T`.
pub trait Push<T> {
    /// Append `t` to `self`.
    fn push(&mut self, t: T);
}

// ----------------------------------------------------------------------------

/// Construct a multi-dimensional collection. This is used to implement
/// [`View::collect()`].
///
/// The collection must implement `View`. The index type will be
/// [`View::I`] and the element type will be [`View::T`].
pub trait NewView: View {
    /// The type of a partially constructed `Self`.
    type Buffer: Push<Self::T>;

    /// Construct a `Self` of size `size`.
    ///
    /// - callback - This will be called once, passing a `Self::Buffer` large
    /// enough to hold `size` items. It must fill the buffer by calling
    /// `Push::push()` once for each item. The items must be pushed in the
    /// order defined by `<Self::I as Index>::to_usize()`.
    ///
    /// # Panics
    ///
    /// Panics if `callback` does, or if it pushes the wrong number of items.
    fn new_view(
        size: <Self::I as Index>::Size,
        callback: impl FnOnce(&mut Self::Buffer),
    ) -> Self;
}

// ----------------------------------------------------------------------------

/// Implemented by types that behave like an array of `Self::T`s indexed by
/// `Self::I`, but whose array elements are computed on demand.
///
/// ### Memory-backed `View`s.
///
/// All implementations of `View` defined in this crate implement
/// [`MemoryView`] when they are backed by memory, i.e. when [`View::at()`]
/// clones a value in memory. For example, a `View` which wraps another `View`
/// and merely manipulates its [`Index`] will implement `MemoryView`.
///
/// You are encouraged to define your `View`s similarly, e.g. using the macro
/// [`impl_memoryview`] which is provided for this purpose.
///
/// ### Arithmetic
///
/// All implementations of `View` defined in this crate define the standard
/// Rust arithmetic operators to mean pointwise arithmetic. For example,
/// `(v + w).at(i)` gives the same answer as `v.at(i) + w.at(i)`.
///
/// More generally, `View`s with different index types can be added if they are
/// compatible according to the [`Broadcast`] trait. For example,
/// `(v + Scalar(x)).at(i)` gives the same answer as `v.at(i) + x`.
///
/// You are encouraged to define your `View`s similarly, e.g. using the macro
/// [`impl_ops_for_view`] which is provided for this purpose.
///
/// ### Ownership
///
/// If `V` implements `View`, then so do `&V`, `Box<V>`, `Rc<V>` and all other
/// types that [`Deref`] to `V`. This means that the `View` structures you can
/// build are agnostic about the ownership of the data they access.
///
/// ```
/// use multidimension::{Index, View, Array};
/// let a: Array<_, _> = std::rc::Rc::new(usize::all(5)).collect();
/// assert_eq!(a.as_ref(), [0, 1, 2, 3, 4]);
/// ```
///
/// If you implement a `View` that draws data from another `View`, it should
/// own the other view, as this is the most general design.
///
/// ```
/// use multidimension::{Index, View};
/// /// `MyView` owns a `V`.
/// struct MyView<V>(V);
/// impl<V: View> View for MyView<V> {
///     type I = V::I;
///     type T = V::T;
///     fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
///     fn at(&self, index: Self::I) -> Self::T { self.0.at(index) }
/// }
/// /// MyView can nonetheless borrow a `V`.
/// fn my_borrow<V: View>(v: &V) -> MyView<&V> { MyView(v) }
/// ```
///
/// [`impl_memoryview`]: super::impl_memoryview
pub trait View: Sized {
    /// The index type.
    type I: Index;

    /// The element type.
    type T: Clone;

    /// The size of the array.
    fn size(&self) -> <Self::I as Index>::Size;

    /// The number of elements in `Self`.
    fn len(&self) -> usize { <Self::I as Index>::length(self.size()) }

    /// Compute the element at `index`.
    fn at(&self, index: Self::I) -> Self::T;

    /// Materialises this `View` into a collection of type `A`, e.g. an
    /// [`Array`].
    ///
    /// This method guarantees to call [`self.at()`] exactly once for each
    /// index.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = usize::all(5).collect();
    /// assert_eq!(a.as_ref(), [0, 1, 2, 3, 4]);
    /// ```
    ///
    /// [`Array`]: super::Array
    /// [`self.at()`]: Self::at()
    fn collect<A>(&self) -> A where
        A: NewView<I=Self::I, T=Self::T>,
    {
        A::new_view(self.size(), |buffer| { self.each(|t| buffer.push(t)); })
    }

    /// Creates a `View` that represents this nested `View`.
    ///
    /// [`Self::T`] must implement `View` and must have a static size. If
    /// `Self` is indexed by `Foo` and `Self::T` is indexed by `Bar` then the
    /// result will be indexed by `(Foo, Bar)`.
    ///
    /// `Self` must implement `MemoryView`. Otherwise, this method would be too
    /// expensive, and you should instead call `nested_collect()` to store all
    /// its elements into some kind of collection.
    ///
    /// ```
    /// use multidimension::{Index, View, MemoryView, Scalar, Array};
    /// let mut a = Array::new(3, [
    ///     Scalar("apple"), Scalar("body"), Scalar("crane"),
    /// ]);
    /// (&mut a).nested()[(1, ())] = "BODY";
    /// ```
    fn nested(self) -> Nested<Self> where
        Self: MemoryView,
        Self::T: View,
        <Self::T as View>::I: Index<Size=()>,
    {
        Nested(self)
    }

    /// Materialises this nested `View` into a collection of type `A`, e.g. an
    /// [`Array`].
    ///
    /// [`Self::T`] must implement `View`. If `Self` is indexed by `Foo` and
    /// `Self::T` is indexed by `Bar` then `A` will be indexed by `(Foo, Bar`).
    ///
    /// Unlike `self.nested().collect()`, this method guarantees to call
    /// [`self.at()`] exactly once for each index, does not require `Self` to
    /// implement `MemoryView`, and does not require `size` to be `()`. When
    /// both are possible, they produce the same result.
    ///
    /// # Panics
    ///
    /// Panics if any `Self::T` returned by `self.at()` has a [`size()`] that
    /// differs from `size`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    ///
    /// // `Behead(V)` hides the first element of a one-dimensional `View` `V`.
    /// #[derive(Clone)]
    /// pub struct Behead<V: View<I=usize>>(V);
    /// impl<V: View<I=usize>> View for Behead<V> {
    ///     type I = usize;
    ///     type T = V::T;
    ///     fn size(&self) -> usize { self.0.size() - 1 }
    ///     fn at(&self, index: usize) -> Self::T { self.0.at(index + 1) }
    /// }
    ///
    /// // Apply `Behead` to every row of a two-dimensional `View`.
    /// let a: Array<(bool, usize), _> = <(bool, usize)>::all(4).collect();
    /// assert_eq!(a.as_ref(), [
    ///     (false, 0), (false, 1), (false, 2), (false, 3),
    ///     (true , 0), (true , 1), (true , 2), (true , 3),
    /// ]);
    /// let a_beheaded: Array<(bool, usize), _> = a.rows().map(Behead).nested_collect(3);
    /// assert_eq!(a_beheaded.as_ref(), [
    ///     (false, 1), (false, 2), (false, 3),
    ///     (true , 1), (true , 2), (true , 3),
    /// ]);
    /// ```
    ///
    /// [`Array`]: super::Array
    /// [`self.at()`]: Self::at()
    /// [`size()`]: View::size()
    fn nested_collect<A>(
        self,
        size: <<Self::T as View>::I as Index>::Size,
    ) -> A where
        Self::T: View,
        A: NewView<I=(Self::I, <Self::T as View>::I), T=<Self::T as View>::T>,
    {
        A::new_view((self.size(), size), |buffer| {
            self.each(|v| {
                assert_eq!(v.size(), size);
                v.each(|t| buffer.push(t));
            });
        })
    }

    /// Apply `f` to every element of this `View` in turn.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = usize::all(5).collect();
    /// let mut total = 0;
    /// a.each(|x| { total += x; });
    /// assert_eq!(total, 10);
    /// ```
    fn each(self, mut f: impl FnMut(Self::T)) {
        Self::I::each(self.size(), |i| f(self.at(i)));
    }

    /// Creates a `View` with the same `Index` type as `self` such that `at(i)`
    /// returns `(i, self.at(i))`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<usize, &str> = Array::new(3, ["apple", "body", "crane"]);
    /// let ea: Array<usize, (usize, &str)> = a.enumerate().collect();
    /// assert_eq!(ea.as_ref(), [
    ///     (0, "apple"),
    ///     (1, "body"),
    ///     (2, "crane"),
    /// ]);
    ///```
    fn enumerate(self) -> Enumerate<Self> {
        Enumerate(self)
    }

    /// Returns a `View` that maps `(i, i)` to `t` when `Self` maps `i` to `t`.
    /// It maps `(i, j)` to `zero` if `i != j`.
    ///
    /// `zero` can be any value of type `V::T`. It doesn't have to be zero.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = usize::all(3).map(|x| x + 10).diagonal(0).collect();
    /// assert_eq!(a.as_ref(), [
    ///     10, 0, 0,
    ///     0, 11, 0,
    ///     0, 0, 12,
    /// ]);
    /// ```
    fn diagonal(self, zero: Self::T) -> Diagonal<Self> {
        Diagonal(self, zero)
    }

    /// Creates a `View` that applies `f` to the elements of `Self`.
    ///
    /// There is no guarantee that the elements will be passed to `f` in a
    /// particular order, only once, or at all.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = usize::all(5).map(|x| x*x).collect();
    /// assert_eq!(a.as_ref(), [0, 1, 4, 9, 16]);
    /// ```
    fn map<U: Clone, F>(self, f: F) -> Map<Self, F> where
        F: Fn(Self::T) -> U,
    {
        Map(self, f)
    }

    /// Creates a `View` that uses `self` to select elements of `other`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<bool, usize> = Array::new((), [2, 1]);
    /// let b: Array<usize, &str> = Array::new(3, ["apple", "body", "crane"]);
    /// let ab: Array<bool, &str> = a.compose(b).collect();
    /// assert_eq!(ab.as_ref(), ["crane", "body"]);
    /// ```
    fn compose<V: View<I=Self::T>>(self, other: V) -> Compose<Self, V> {
        Compose(self, other)
    }

    /// Concatenates `self` with `other` along an axis of type `usize`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<usize, &str> = Array::new(2, ["apple", "body"]);
    /// let b: Array<usize, &str> = Array::new(2, ["crane", "dump"]);
    /// let ab: Array<usize, &str> = a.concat::<_, (), ()>(b).iso().collect();
    /// assert_eq!(ab.as_ref(), ["apple", "body", "crane", "dump"]);
    /// ```
    fn concat<V: View<T=Self::T>, I: Index, J: Index>(self, other: V)
    -> Concat<Self, V, I, J> where
        Self::I: Isomorphic<(I, usize, J)>,
        <Self::I as Index>::Size: Isomorphic<(I::Size, usize, J::Size)>,
        V::I: Isomorphic<(I, usize, J)>,
        <V::I as Index>::Size: Isomorphic<(I::Size, usize, J::Size)>,
    {
        let (self_i, self_size, self_j) = self.size().to_iso();
        let (other_i, _other_size, other_j) = other.size().to_iso();
        assert_eq!(self_i, other_i);
        assert_eq!(self_j, other_j);
        Concat(self, other, self_size, PhantomData)
    }

    /// Replace an axis [`Index`]ed by `usize` with one indexed by `X`.
    ///
    /// - `from_length` - a function to compute the `X::Size` of the new axis,
    ///   given the `usize::Size` of the old one.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<(usize, usize, usize), &str> = Array::new((3, 2, 1), ["A", "a", "B", "b", "C", "c"]);
    /// let b: Array<(usize, bool, usize), &str> = (&a).from_usize::<usize, bool, usize>(|_| ()).collect();
    /// assert_eq!(b.at((2, true, 0)), a.at((2, 1, 0)));
    /// ```
    fn from_usize<I: Index, X: Index, J: Index>(
        self,
        from_length: impl FnOnce(usize) -> X::Size,
    ) -> FromUsize<Self, I, X, J> where
        Self::I: Isomorphic<(I, usize, J)>,
        <Self::I as Index>::Size: Isomorphic<<(I, usize, J) as Index>::Size>,
    {
        let (_, old_size, _) = self.size().to_iso();
        let size = from_length(old_size);
        assert_eq!(X::length(size), old_size);
        FromUsize(self, size, PhantomData)
    }

    /// Replace an axis [`Index`]ed by `X` with one indexed by `usize`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<(usize, bool, usize), &str> = Array::new((3, (), 1), ["A", "a", "B", "b", "C", "c"]);
    /// let b: Array<(usize, usize, usize), &str> = (&a).to_usize::<usize, bool, usize>().collect();
    /// assert_eq!(b.at((2, 1, 0)), a.at((2, true, 0)));
    /// ```
    fn to_usize<I: Index, X: Index, J: Index>(self) -> ToUsize<Self, I, X, J> where
        Self::I: Isomorphic<(I, X, J)>,
        <Self::I as Index>::Size: Isomorphic<<(I, X, J) as Index>::Size>,
    {
        ToUsize(self, PhantomData)
    }

    /// Insert an axis of type `J` and length `1`.
    ///
    /// - size - the size of the new axis. `J::length(size)` must be `1`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<(bool, bool), &str> = Array::new(((), ()), ["A", "a", "B", "b"]);
    /// let b: Array<(bool, usize, bool), &str> = a.insert_one::<bool, usize, bool>(1).collect();
    /// assert_eq!(b.size(), ((), 1, ()));
    /// assert_eq!(b.as_ref(), ["A", "a", "B", "b"]);
    /// ```
    fn insert_one<I: Index, J: Index, K: Index>(self, size: J::Size) -> InsertOne<Self, I, J, K> where
        Self::I: Isomorphic<(I, K)>,
        <Self::I as Index>::Size: Isomorphic<(I::Size, K::Size)>,
    {
        assert_eq!(J::length(size), 1);
        InsertOne(self, size, PhantomData)
    }

    /// Remove an axis of type `J` and length `1`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<(bool, usize, bool), &str> = Array::new(((), 1, ()), ["A", "a", "B", "b"]);
    /// let b: Array<(bool, bool), &str> = a.remove_one::<bool, usize, bool>().collect();
    /// assert_eq!(b.size(), ((), ()));
    /// assert_eq!(b.as_ref(), ["A", "a", "B", "b"]);
    /// ```
    fn remove_one<I: Index, J: Index, K: Index>(self) -> RemoveOne<Self, I, J, K> where
        Self::I: Isomorphic<(I, J, K)>,
        <Self::I as Index>::Size: Isomorphic<(I::Size, J::Size, K::Size)>,
    {
        let (_, j_size, _) = self.size().to_iso();
        assert_eq!(J::length(j_size), 1);
        let (q, j) = J::from_usize(j_size, 0);
        assert_eq!(q, 0);
        assert_eq!(j.to_usize(j_size), 0);
        RemoveOne(self, j, PhantomData)
    }

    /// Creates a view such that `at((i, x, j))` gives
    /// `self.at((i, other.at(x), j))`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<usize, usize> = Array::new(2, [2, 1]);
    /// let b: Array<(bool, usize), &str> = Array::new(3, [
    ///     "apple", "body", "crane",
    ///     "APPLE", "BODY", "CRANE",
    /// ]);
    /// let ab: Array<(bool, usize, ()), &str> = b.map_axis(a).collect();
    /// assert_eq!(ab.as_ref(), [
    ///     "crane", "body",
    ///     "CRANE", "BODY",
    /// ]);
    /// ```
    fn map_axis<I: Index, V: View, J: Index>(self, other: V) -> MapAxis<Self, I, V, J> where
        V::T: Index,
        Self::I: Isomorphic<(I, V::T, J)>,
        <Self::I as Index>::Size: Isomorphic<(I::Size, <V::T as Index>::Size, J::Size)>,
    {
        MapAxis(self, PhantomData, other, PhantomData)
    }

    /// Creates a `View` that pairs of an element of `self` and an element of
    /// `other`.
    ///
    /// Two `View`s can be zipped only if they have compatible indices. See
    /// trait [`Broadcast`] for more details.
    ///
    /// Here's an example of zipping two 1D [`Array`]s of the same size:
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<usize, usize> = usize::all(3).collect();
    /// let b: Array<usize, &str> = Array::new(3, ["apple", "body", "crane"]);
    /// let ab: Array<usize, (usize, &str)> = a.zip(b).collect();
    /// assert_eq!(ab.as_ref(), [
    ///     (0, "apple"),
    ///     (1, "body"),
    ///     (2, "crane"),
    /// ]);
    /// ```
    ///
    /// Here's an example of zipping a 1D [`Array`] with a [`Scalar`]:
    /// ```
    /// use multidimension::{Index, View, Array, Scalar};
    /// let a: Array<usize, usize> = usize::all(3).collect();
    /// let b = Scalar("repeated");
    /// let ab: Array<usize, (usize, &str)> = a.zip(b).collect();
    /// assert_eq!(ab.as_ref(), [
    ///     (0, "repeated"),
    ///     (1, "repeated"),
    ///     (2, "repeated"),
    /// ]);
    /// ```
    /// 
    /// Here's an example of zipping two 2D [`Array`]s of different shape:
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<(usize, ()), usize> = usize::all(3).iso().collect();
    /// let b: Array<((), bool), bool> = bool::all(()).iso().collect();
    /// let ab: Array<(usize, bool), (usize, bool)> = a.zip(b).collect();
    /// assert_eq!(ab.as_ref(), [
    ///     (0, false), (0, true),
    ///     (1, false), (1, true),
    ///     (2, false), (2, true),
    /// ]);
    /// ```
    ///
    /// [`Array`]: super::Array
    fn zip<V: View>(self, other: V) -> Zip<Self, V, super::ops::Pair> where
        Self::I: Broadcast<V::I>,
    {
        Zip(self, other, PhantomData)
    }

    /// Creates a `View` that pairs of an element of `self` and an element of
    /// `other`, then applies binary operator `B`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array, ops::Add};
    /// let a: Array<usize, usize> = Array::new(3, [9, 8, 7]);
    /// let b: Array<usize, usize> = Array::new(3, [10, 20, 30]);
    /// // Equivalent to `(a + b).collect()`.
    /// let ab: Array<usize, usize> = a.binary::<_, Add>(b).collect();
    /// assert_eq!(ab.as_ref(), [19, 28, 37]);
    /// ```
    fn binary<V: View, B>(self, other: V) -> Zip<Self, V, B> where
        Self::I: Broadcast<V::I>,
        B: Binary<Self::T, V::T>,
    {
        Zip(self, other, PhantomData)
    }

    /// Wrap some of the [`Index`]es of this `View` in [`Coated`].
    ///
    /// Coating `Index` types prevents [`Isomorphic`] from looking inside them.
    /// This is mostly useful for generic functions that would not otherwise
    /// compile, because they don't know at compile time the internal structure
    /// of their generic type parameters.
    ///
    /// [`Coated`]: super::Coated
    ///
    /// ```
    /// use multidimension::{Index, View, Array, Coated};
    ///
    /// /// Halve the length of the second axis by pairing up the elements.
    /// fn group_pairs<I: Index, T: Clone>(
    ///     view: impl View<I=(I, usize), T=T>,
    /// ) -> impl View<I=(I, usize, bool), T=T> {
    ///     let view = view.coat::<(Coated<I>, usize)>();
    ///     // Next two lines would not compile without the `Coated` wrapper.
    ///     let view = view.from_usize::<Coated<I>, (usize, bool), ()>(|length| (length / 2, ()));
    ///     let view = view.iso::<(Coated<I>, usize, bool)>();
    ///     let view = view.coat::<(I, usize, bool)>();
    ///     view
    /// }
    ///
    /// let a: Array<(usize, usize), &str> = Array::new((2, 6), [
    ///     "a", "b", "c", "d", "e", "f",
    ///     "A", "B", "C", "D", "E", "F",
    /// ]);
    /// let b: Array<(usize, usize, bool), &str> = group_pairs(a).collect();
    /// assert_eq!(b.size(), (2, 3, ()));
    /// assert_eq!(b.as_ref(), [
    ///     "a", "b",   "c", "d",   "e", "f",
    ///     "A", "B",   "C", "D",   "E", "F",
    /// ]);
    /// ```
    fn coat<I: Index>(self) -> Coat<Self, I> where
        I: super::Coat<Self::I>,
        Self::I: super::Coat<I>,
        I::Size: super::Coat<<Self::I as Index>::Size>,
        <Self::I as Index>::Size: super::Coat<I::Size>,
    {
        Coat(self, PhantomData)
    }

    /// Change the index type of this `View` to an [`Isomorphic`] type.
    fn iso<J: Index>(self) -> Iso<Self, J> where
        J: Isomorphic<Self::I>,
        J::Size: Isomorphic<<Self::I as Index>::Size>,
    {
        Iso(self, PhantomData)
    }

    /// Reorder the indices of this `View` up to [`Isomorphic`].
    ///
    /// This method makes a `View` indexed by `(I, (X, Y), J)` if `Self` is
    /// indexed by `(I, (Y, X), J)`, i.e. it swaps the `X` and `Y` indices.
    /// Note that `I` and/or `J` can be `()` if they are unwanted.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = <(usize, usize)>::all((3, 2)).collect();
    /// assert_eq!(a.as_ref(), [
    ///     (0, 0), (0, 1),
    ///     (1, 0), (1, 1),
    ///     (2, 0), (2, 1),
    /// ]);
    /// let b: Array<_, _> = a.transpose::<(), usize, usize, ()>().collect();
    /// assert_eq!(b.as_ref(), [
    ///     (0, 0), (1, 0), (2, 0),
    ///     (0, 1), (1, 1), (2, 1),
    /// ]);
    /// ```
    fn transpose<I: Index, X: Index, Y: Index, J: Index>(self)
    -> Transpose<Self, I, X, Y, J> where
        (I, (Y, X), J): Isomorphic<Self::I>,
        (I::Size, (Y::Size, X::Size), J::Size): Isomorphic<<Self::I as Index>::Size>,
    {
        Transpose(self, PhantomData)
    }

    /// Returns a `View` whose `at(j)` returns `self.at(i, j)`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = <(usize, usize)>::all((3, 2)).collect();
    /// assert_eq!(a.as_ref(), [
    ///     (0, 0), (0, 1),
    ///     (1, 0), (1, 1),
    ///     (2, 0), (2, 1),
    /// ]);
    /// let a_row: Array<_, _> = a.row::<usize, usize>(1).collect();
    /// assert_eq!(a_row.as_ref(), [
    ///     (1, 0), (1, 1),
    /// ]);
    /// ```
    fn row<I: Index, J: Index>(self, i: I) -> Row<Self, I, J> where
        (I, J): Isomorphic<Self::I>,
        (I::Size, J::Size): Isomorphic<<Self::I as Index>::Size>,
    {
        Row(self, i, PhantomData)
    }

    /// Returns a `View` whose `at(i)` returns `self.row(i)`.
    fn rows<I: Index, J: Index>(&self) -> Rows<&Self, I, J> where
        (I, J): Isomorphic<Self::I>,
        (I::Size, J::Size): Isomorphic<<Self::I as Index>::Size>,
    {
        Rows(self, PhantomData)
    }

    /// Returns a `View` whose `at(i)` returns `self.at(i, j)`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = <(usize, usize)>::all((3, 2)).collect();
    /// assert_eq!(a.as_ref(), [
    ///     (0, 0), (0, 1),
    ///     (1, 0), (1, 1),
    ///     (2, 0), (2, 1),
    /// ]);
    /// let a_column: Array<_, _> = a.column::<usize, usize>(1).collect();
    /// assert_eq!(a_column.as_ref(), [
    ///     (0, 1), (1, 1), (2, 1),
    /// ]);
    /// ```
    fn column<I: Index, J: Index>(self, j: J) -> Column<Self, I, J> where
        (I, J): Isomorphic<Self::I>,
        (I::Size, J::Size): Isomorphic<<Self::I as Index>::Size>,
    {
        Column(self, PhantomData, j)
    }

    /// Returns a `View` whose `at(j)` returns `self.column(j)`.
    fn columns<I: Index, J: Index>(&self) -> Columns<&Self, I, J> where
        (I, J): Isomorphic<Self::I>,
        (I::Size, J::Size): Isomorphic<<Self::I as Index>::Size>,
    {
        Columns(self, PhantomData)
    }
}

impl<V: View, T: Deref<Target=V>> View for T {
    type I = V::I;
    type T = V::T;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { V::size(self) }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { V::at(self, index) }
}

// ----------------------------------------------------------------------------

/// A [`View`] that is backed by memory.
///
/// [`self.at(index)`] must be equivalent to `self.at_ref(index).clone()`.
/// `self.at_ref(index)` must be equivalent to `&*self.at_mut(index)`.
///
/// You might find the macro [`impl_memoryview`] helpful when implementing this
/// trait.
///
/// ### Relationship to [`std::ops::Index`]
///
/// All implementations of `MemoryView` defined in this crate implement
/// [`std::ops::Index`] and [`std::ops::IndexMut`] to mean the same as
/// [`at_ref()`] and [`at_mut()`]. In particular, the index type is [`Self::I`]
/// and the [`Output`] type is [`Self::T`].
///
/// You are encouraged to define your own `MemoryView`s similarly.
///
/// Note, however, that some types (e.g. `&V`, `&mut V` and `Box<V>` where `V`
/// implement `MemoryView`) implement `MemoryView` but not `Index` or
/// `IndexMut`. This is unfortunate, but it is impossible for this crate to fix
/// it. Therefore, in generic code, you can only rely on `MemoryView` methods.
///
/// [`self.at(index)`]: View::at()
/// [`Output`]: std::ops::Index::Output
/// [`impl_memoryview`]: super::impl_memoryview
pub trait MemoryView: View {
    /// Borrow the element at `index`.
    fn at_ref(&self, index: Self::I) -> &Self::T;

    /// Mutably borrow the element at `index`.
    fn at_mut(&mut self, index: Self::I) -> &mut Self::T;
}

impl<T: DerefMut> MemoryView for T where <T as Deref>::Target: MemoryView {
    #[inline(always)]
    fn at_ref(&self, index: Self::I) -> &Self::T { (**self).at_ref(index) }
    #[inline(always)]
    fn at_mut(&mut self, index: Self::I) -> &mut Self::T { (**self).at_mut(index) }
}

/// A helper macro for implementing [`MemoryView`].
///
/// The macro defines a private method `inner_index()` and implements
/// [`std::ops::Index`], [`std::ops::IndexMut`] and [`MemoryView`]
///
/// ```
/// use multidimension::{Index, View, MemoryView, impl_memoryview};
///
/// struct Reversed<V: View>(V);
///
/// impl_memoryview!(Reversed<V: View<I=usize>> where
///     V: MemoryView,
/// {
///     |self_, index| (self_.0)[self_.size() - 1 - index]
/// });
///
/// impl<V: View<I=usize>> View for Reversed<V> {
///     type I = usize;
///     type T = V::T;
///     fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
///     fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
/// }
/// ```
#[macro_export]
macro_rules! impl_memoryview {
    ($v:ident<
        $($a: lifetime,)?
        $($param:ident$(: $bound:path)?),*
    > where
        $inner:ty: MemoryView,
        $($where_type:ty: $where_bound:path),*
    {
        |$self:ident, $index:ident| ($collection_expr:expr)[$index_expr:expr]
    }) => {
        impl<
            $($a,)?
            $($param$(: $bound)?),*
        > $v<$($a,)? $($param),*> where
            $($where_type: $where_bound),*
        {
            /// Map a `Self::I` to a `$inner::I`.
            #[inline(always)]
            fn inner_index(&self, $index: <Self as $crate::View>::I) -> <$inner as $crate::View>::I {
                #[allow(unused)]
                let $self = self;
                $index_expr
            }
        }

        impl<
            $($a,)?
            $($param$(: $bound)?),*
        > $crate::MemoryView for $v<$($a,)? $($param),*> where
            $inner: $crate::MemoryView,
            $($where_type: $where_bound),*
        {
            fn at_ref(&self, $index: Self::I) -> &Self::T {
                let $self = self;
                let index = $self.inner_index($index);
                $collection_expr.at_ref(index)
            }

            fn at_mut(&mut self, $index: Self::I) -> &mut Self::T {
                let $self = self;
                let index = $self.inner_index($index);
                $collection_expr.at_mut(index)
            }
        }

        $crate::impl_ops_for_memoryview!($v<$($a,)? $($param$(: $bound)?),*>);
    }
}

// ----------------------------------------------------------------------------

/// The return type of [`View::view_ref()`].
pub struct Nested<V>(V);

impl<V: MemoryView> View for Nested<V> where
    V::T: View,
    <V::T as View>::I: Index<Size=()>,
{
    type I = (V::I, <V::T as View>::I);
    type T = <V::T as View>::T;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { (self.0.size(), ()) }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at_ref(index.0).at(index.1) }
}

impl<V: MemoryView> MemoryView for Nested<V> where
    V::T: MemoryView,
    <V::T as View>::I: Index<Size=()>,
{
    #[inline(always)]
    fn at_ref(&self, index: Self::I) -> &Self::T { self.0.at_ref(index.0).at_ref(index.1) }
    #[inline(always)]
    fn at_mut(&mut self, index: Self::I) -> &mut Self::T { self.0.at_mut(index.0).at_mut(index.1) }
}

impl_ops_for_view!(Nested<V>);
impl_ops_for_memoryview!(Nested<V>);

// ----------------------------------------------------------------------------

/// The return type of [`View::enumerate()`].
#[derive(Debug, Copy, Clone)]
pub struct Enumerate<V>(V);

impl<V: View> View for Enumerate<V> {
    type I = V::I;
    type T = (V::I, V::T);
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { (index, self.0.at(index)) }
}

impl_ops_for_view!(Enumerate<V>);

// ----------------------------------------------------------------------------

/// The return type of [`View::diagonal()`].
#[derive(Debug, Copy, Clone)]
pub struct Diagonal<V: View>(V, V::T);

impl<V: View> View for Diagonal<V> {
    type I = (V::I, V::I);
    type T = V::T;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { (self.0.size(), self.0.size()) }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T {
        if index.0 == index.1 { self.0.at(index.0) } else { self.1.clone() }
    }
}

impl<V: View> MemoryView for Diagonal<V> where V: MemoryView {
    #[inline(always)]
    fn at_ref(&self, index: Self::I) -> &Self::T {
        if index.0 == index.1 { self.0.at_ref(index.0) } else { &self.1 }
    }

    #[inline(always)]
    fn at_mut(&mut self, index: Self::I) -> &mut Self::T {
        if index.0 == index.1 { self.0.at_mut(index.0) } else { &mut self.1 }
    }
}

impl_ops_for_view!(Diagonal<V: View>);
impl_ops_for_memoryview!(Diagonal<V: View>);

// ----------------------------------------------------------------------------

/// The return type of [`View::map()`].
#[derive(Debug, Copy, Clone)]
pub struct Map<V, F>(V, F);

impl<V: View, U: Clone, F: Fn(V::T) -> U> View for Map<V, F> {
    type I = V::I;
    type T = U;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.1(self.0.at(index)) }
}

impl_ops_for_view!(Map<V, F>);

// ----------------------------------------------------------------------------

/// The return type of [`View::compose()`].
#[derive(Debug, Copy, Clone)]
pub struct Compose<V, W>(V, W);

impl<V: View, W: View<I=V::T>> View for Compose<V, W> {
    type I = V::I;
    type T = W::T;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.1.at(self.inner_index(index)) }
}

impl_memoryview!(Compose<V: View, W: View<I=V::T>> where
    W: MemoryView,
{
    |self_, index| (self_.1)[self_.0.at(index)]
});

impl_ops_for_view!(Compose<V, W>);

// ----------------------------------------------------------------------------

/// The return type of [`View::concat()`].
#[derive(Debug, Copy, Clone)]
pub struct Concat<V, W, I, J>(V, W, usize, PhantomData<(I, J)>);

impl<V: View, W: View<T=V::T>, I: Index, J: Index> View for Concat<V, W, I, J> where
    V::I: Isomorphic<(I, usize, J)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, usize, J::Size)>,
    W::I: Isomorphic<(I, usize, J)>,
    <W::I as Index>::Size: Isomorphic<(I::Size, usize, J::Size)>,
{
    type I = (I, usize, J);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        let (i, w_size, j) = self.1.size().to_iso();
        (i, self.2 + w_size, j)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T {
        let (i, index, j) = index;
        if index < self.2 {
            self.0.at(Isomorphic::from_iso((i, index, j)))
        } else {
            self.1.at(Isomorphic::from_iso((i, index - self.2, j)))
        }
    }
}

impl<V: View, W: View<T=V::T>, I: Index, J: Index> MemoryView for Concat<V, W, I, J> where
    V: MemoryView,
    W: MemoryView,
    V::I: Isomorphic<(I, usize, J)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, usize, J::Size)>,
    W::I: Isomorphic<(I, usize, J)>,
    <W::I as Index>::Size: Isomorphic<(I::Size, usize, J::Size)>,
{
    #[inline(always)]
    fn at_ref(&self, index: Self::I) -> &Self::T {
        let (i, index, j) = index;
        if index < self.2 {
            self.0.at_ref(Isomorphic::from_iso((i, index, j)))
        } else {
            self.1.at_ref(Isomorphic::from_iso((i, index - self.2, j)))
        }
    }

    #[inline(always)]
    fn at_mut(&mut self, index: Self::I) -> &mut Self::T {
        let (i, index, j) = index;
        if index < self.2 {
            self.0.at_mut(Isomorphic::from_iso((i, index, j)))
        } else {
            self.1.at_mut(Isomorphic::from_iso((i, index - self.2, j)))
        }
    }
}

impl_ops_for_view!(Concat<V, W, I, J>);
impl_ops_for_memoryview!(Concat<V, W, I, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::from_usize()`]
#[derive(Debug, Copy, Clone)]
pub struct FromUsize<V, I, X: Index, J>(V, X::Size, PhantomData<(I, J)>);

impl<V: View, I: Index, X: Index, J: Index> View for FromUsize<V, I, X, J> where
    V::I: Isomorphic<(I, usize, J)>,
    <V::I as Index>::Size: Isomorphic<<(I, usize, J) as Index>::Size>,
{
    type I = (I, X, J);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        let (i_size, _, j_size) = self.0.size().to_iso();
        (i_size, self.1, j_size)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(FromUsize<V: View, I: Index, X: Index, J: Index> where
    V: MemoryView,
    V::I: Isomorphic<(I, usize, J)>,
    <V::I as Index>::Size: Isomorphic<<(I, usize, J) as Index>::Size>
{
    |self_, index| (self_.0)[{
        let (i, x, j) = index;
        V::I::from_iso((i, x.to_usize(self_.1), j))
    }]
});

impl_ops_for_view!(FromUsize<V, I, X: Index, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::to_usize()`]
#[derive(Debug, Copy, Clone)]
pub struct ToUsize<V, I, X, J>(V, PhantomData<(I, X, J)>);

impl<V: View, I: Index, X: Index, J: Index> View for ToUsize<V, I, X, J> where
    V::I: Isomorphic<(I, X, J)>,
    <V::I as Index>::Size: Isomorphic<<(I, X, J) as Index>::Size>,
{
    type I = (I, usize, J);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        let (i_size, x_size, j_size) = self.0.size().to_iso();
        (i_size, X::length(x_size), j_size)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(ToUsize<V: View, I: Index, X: Index, J: Index> where
    V: MemoryView,
    V::I: Isomorphic<(I, X, J)>,
    <V::I as Index>::Size: Isomorphic<<(I, X, J) as Index>::Size>
{
    |self_, index| (self_.0)[{
        let (i, x, j) = index;
        let (q, x) = X::from_usize(self_.0.size().to_iso().1, x);
        assert_eq!(q, 0);
        V::I::from_iso((i, x, j))
    }]
});

impl_ops_for_view!(ToUsize<V, I, X, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::insert_one()`]
#[derive(Debug, Copy, Clone)]
pub struct InsertOne<V, I, J: Index, K>(V, J::Size, PhantomData<(I, K)>);

impl<V: View, I: Index, J: Index, K: Index> View for InsertOne<V, I, J, K> where
    V::I: Isomorphic<(I, K)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, K::Size)>,
{
    type I = (I, J, K);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        let (i, k) = self.0.size().to_iso();
        (i, self.1, k)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(InsertOne<V: View, I: Index, J: Index, K: Index> where
    V: MemoryView,
    V::I: Isomorphic<(I, K)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, K::Size)>
{
    |self_, index| (self_.0)[{
        let (i, j, k) = index;
        assert_eq!(j.to_usize(self_.1), 0);
        Isomorphic::from_iso((i, k))
    }]
});

impl_ops_for_view!(InsertOne<V, I, J: Index, K>);

// ----------------------------------------------------------------------------

/// The return type of [`View::insert_one()`]
#[derive(Debug, Copy, Clone)]
pub struct RemoveOne<V, I, J, K>(V, J, PhantomData<(I, K)>);

impl<V: View, I: Index, J: Index, K: Index> View for RemoveOne<V, I, J, K> where
    V::I: Isomorphic<(I, J, K)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, J::Size, K::Size)>,
{
    type I = (I, K);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        let (i, _, k) = self.0.size().to_iso();
        (i, k)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(RemoveOne<V: View, I: Index, J: Index, K: Index> where
    V: MemoryView,
    V::I: Isomorphic<(I, J, K)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, J::Size, K::Size)>
{
    |self_, index| (self_.0)[{
        let (i, k) = index;
        Isomorphic::from_iso((i, self_.1, k))
    }]
});

impl_ops_for_view!(RemoveOne<V, I, J, K>);

// ----------------------------------------------------------------------------

/// The return type of [`View::map_axis()`].
#[derive(Debug, Copy, Clone)]
pub struct MapAxis<V, I, W, J>(V, PhantomData<I>, W, PhantomData<J>);

impl<V: View, I: Index, W: View, J: Index> View for MapAxis<V, I, W, J> where
    W::T: Index,
    V::I: Isomorphic<(I, W::T, J)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, <W::T as Index>::Size, J::Size)>,
{
    type I = (I, W::I, J);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        let (i_size, _, j_size) = self.0.size().to_iso();
        (i_size, self.2.size(), j_size)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(MapAxis<V: View, I: Index, W: View, J: Index> where
    V: MemoryView,
    W::T: Index,
    V::I: Isomorphic<(I, W::T, J)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, <W::T as Index>::Size, J::Size)>
{
    |self_, index| (self_.0)[{
        let (i, x, j) = index;
        Isomorphic::from_iso((i, self_.2.at(x), j))
    }]
});

impl_ops_for_view!(MapAxis<V, I, W, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::zip()`] and [`View::binary()`].
#[derive(Debug, Copy, Clone)]
pub struct Zip<V, W, B>(V, W, PhantomData<B>);

impl<V: View, W: View, B> View for Zip<V, W, B> where
    V::I: Broadcast<W::I>,
    B: Binary<V::T, W::T>,
    B::Output: Clone,
{
    type I = <V::I as Broadcast<W::I>>::Result;
    type T = <B as Binary<V::T, W::T>>::Output;

    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size {
        <V::I as Broadcast<W::I>>::size(self.0.size(), self.1.size())
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T {
        let (v_index, w_index) = <V::I as Broadcast<W::I>>::index(index);
        B::call(self.0.at(v_index), self.1.at(w_index))
    }
}

impl_ops_for_view!(Zip<V, W, B>);

// ----------------------------------------------------------------------------

/// The return type of [`View::coat()`].
#[derive(Debug, Copy, Clone)]
pub struct Coat<V, I>(V, PhantomData<I>);

impl<V: View, I: Index> View for Coat<V, I> where
    I: super::Coat<V::I>,
    V::I: super::Coat<I>,
    I::Size: super::Coat<<V::I as Index>::Size>,
    <V::I as Index>::Size: super::Coat<I::Size>,
{
    type I = I;
    type T = V::T;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { self.0.size().coat() }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(Coat<V: View, I: Index> where
    V: MemoryView,
    I: super::Coat<V::I>,
    V::I: super::Coat<I>,
    I::Size: super::Coat<<V::I as Index>::Size>,
    <V::I as Index>::Size: super::Coat<I::Size>
{
    |self_, index| (self_.0)[index.coat()]
});

impl_ops_for_view!(Coat<V, I>);

// ----------------------------------------------------------------------------

/// The return type of [`View::iso()`].
#[derive(Debug, Copy, Clone)]
pub struct Iso<V, J: Index>(V, PhantomData<J>);

impl<V: View, J: Index> View for Iso<V, J> where
    J: Isomorphic<V::I>,
    J::Size: Isomorphic<<V::I as Index>::Size>,
{
    type I = J;
    type T = V::T;
    #[inline(always)]
    fn size(&self) -> <Self::I as Index>::Size { Isomorphic::from_iso(self.0.size()) }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(Iso<V: View, J: Index> where
    V: MemoryView,
    J: Isomorphic<V::I>,
    J::Size: Isomorphic<<V::I as Index>::Size>
{
    |self_, index| (self_.0)[index.to_iso()]
});

impl_ops_for_view!(Iso<V, J: Index>);

// ----------------------------------------------------------------------------

/// The return type of [`View::transpose()`].
#[derive(Debug, Copy, Clone)]
pub struct Transpose<V, I, X, Y, J>(V, PhantomData<(I, (X, Y), J)>);

impl<V: View, I: Index, X: Index, Y: Index, J: Index> View for Transpose<V, I, X, Y, J> where
    (I, (Y, X), J): Isomorphic<V::I>,
    (I::Size, (Y::Size, X::Size), J::Size): Isomorphic<<V::I as Index>::Size>,
{
    type I = (I, (X, Y), J);
    type T = V::T;

    #[inline(always)]
    fn size(&self) -> (I::Size, (X::Size, Y::Size), J::Size) {
        let (i, (y, x), j) = Isomorphic::from_iso(self.0.size());
        (i, (x, y), j)
    }

    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(Transpose<V: View, I: Index, X: Index, Y: Index, J: Index> where
    V: MemoryView,
    (I, (Y, X), J): Isomorphic<V::I>,
    (I::Size, (Y::Size, X::Size), J::Size): Isomorphic<<V::I as Index>::Size>
{
    |self_, index| (self_.0)[{
        let (i, (x, y), j) = index;
        (i, (y, x), j).to_iso()
    }]
});

impl_ops_for_view!(Transpose<V, I, X, Y, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::row()`].
#[derive(Debug, Copy, Clone)]
pub struct Row<V, I, J>(V, I, PhantomData<J>);

impl<V: View, I: Index, J: Index> View for Row<V, I, J> where
    (I, J): Isomorphic<V::I>,
    (I::Size, J::Size): Isomorphic<<V::I as Index>::Size>,
{
    type I = J;
    type T = V::T;
    #[inline(always)]
    fn size(&self) -> J::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).1 }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(Row<V: View, I: Index, J: Index> where
    V: MemoryView,
    (I, J): Isomorphic<V::I>,
    (I::Size, J::Size): Isomorphic<<V::I as Index>::Size>
{
    |self_, index| (self_.0)[(self_.1, index).to_iso()]
});

impl_ops_for_view!(Row<V, I, J>);

// ----------------------------------------------------------------------------

/// The return type of `View::rows()`.
#[derive(Debug, Copy, Clone)]
pub struct Rows<V, I, J>(V, PhantomData<(I, J)>);

impl<V: Copy + View, I: Index, J: Index> View for Rows<V, I, J> where
    (I, J): Isomorphic<V::I>,
    (I::Size, J::Size): Isomorphic<<V::I as Index>::Size>,
{
    type I = I;
    type T = Row<V, I, J>;
    #[inline(always)]
    fn size(&self) -> I::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).0 }
    #[inline(always)]
    fn at(&self, index: I) -> Self::T { self.0.row(index) }
}

impl_ops_for_view!(Rows<V, I, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::column()`]
#[derive(Debug, Copy, Clone)]
pub struct Column<V, I, J>(V, PhantomData<I>, J);

impl<V: View, I: Index, J: Index> View for Column<V, I, J> where
    (I, J): Isomorphic<V::I>,
    (I::Size, J::Size): Isomorphic<<V::I as Index>::Size>,
{
    type I = I;
    type T = V::T;
    #[inline(always)]
    fn size(&self) -> I::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).0 }
    #[inline(always)]
    fn at(&self, index: Self::I) -> Self::T { self.0.at(self.inner_index(index)) }
}

impl_memoryview!(Column<V: View, I: Index, J: Index> where
    V: MemoryView,
    (I, J): Isomorphic<V::I>,
    (I::Size, J::Size): Isomorphic<<V::I as Index>::Size>
{
    |self_, index| (self_.0)[(index, self_.2).to_iso()]
});

impl_ops_for_view!(Column<V, I, J>);

// ----------------------------------------------------------------------------

/// The return type of `View::columns()`.
#[derive(Debug, Copy, Clone)]
pub struct Columns<V, I, J>(V, PhantomData<(I, J)>);

impl<V: Copy + View, I: Index, J: Index> View for Columns<V, I, J> where
    (I, J): Isomorphic<V::I>,
    (I::Size, J::Size): Isomorphic<<V::I as Index>::Size>,
{
    type I = J;
    type T = Column<V, I, J>;
    #[inline(always)]
    fn size(&self) -> J::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).1 }
    #[inline(always)]
    fn at(&self, index: J) -> Self::T { self.0.column(index) }
}

impl_ops_for_view!(Columns<V, I, J>);

// ----------------------------------------------------------------------------

/// A 0-dimensional [`View`].
#[derive(Default, Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct Scalar<T: Clone>(pub T);

impl<T: Clone> View for Scalar<T> {
    type I = ();
    type T = T;
    #[inline(always)]
    fn size(&self) -> () { () }
    #[inline(always)]
    fn at(&self, _: ()) -> T { self.0.clone() }
}

impl<T: Clone> MemoryView for Scalar<T> {
    #[inline(always)]
    fn at_ref(&self, _: Self::I) -> &Self::T { &self.0 }
    #[inline(always)]
    fn at_mut(&mut self, _: Self::I) -> &mut Self::T { &mut self.0 }
}

impl_ops_for_view!(Scalar<T: Clone>);
impl_ops_for_memoryview!(Scalar<T: Clone>);

// ----------------------------------------------------------------------------

/// Construct a `View` of size `size` from a function.
///
/// Consider also [`Array::from_fn()`].
///
/// [`Array::from_fn()`]: super::Array::from_fn
///
/// ```
/// use multidimension::{Index, View, fn_view, Array};
/// let a: Array<usize, _> = fn_view(10, |x| x % 3 == 0).collect();
/// assert_eq!(a.as_ref(), [true, false, false, true, false, false, true, false, false, true]);
/// ```
pub fn fn_view<I: Index, T: Clone, F>(size: impl Isomorphic<I::Size>, f: F) -> FnView<I, F> where
    F: Fn(I) -> T,
{
    I::all(size).map(f)
}

/// The return type of [`fn_view()`].
pub type FnView<I, F> = Map<super::All<I>, F>;
