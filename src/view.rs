use std::fmt::{Debug};
use std::marker::{PhantomData};
use std::ops::{Deref};

use super::{Isomorphic, Flatten, Index, Broadcast, impl_ops_for_view, Binary};

/// Implemented by types that behave like an array of `Self::T`s indexed by
/// `Self::I`, but whose array elements are computed on demand.
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
pub trait View: Sized {
    /// The index type.
    type I: Index;

    /// The element type.
    type T;

    /// The size of the array.
    fn size(&self) -> <Self::I as Index>::Size;

    /// The number of elements in `Self`.
    fn len(&self) -> usize { <Self::I as Index>::length(self.size()) }

    /// Compute the element at `index`.
    fn at(&self, index: Self::I) -> Self::T;

    /// Materialises this `View` into a collection of type `A`, e.g. an
    /// [`Array`].
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = usize::all(5).collect();
    /// assert_eq!(a.as_ref(), [0, 1, 2, 3, 4]);
    /// ```
    ///
    /// [`Array`]: super::Array
    fn collect<A: FromView<Self::I, Self::T>>(&self) -> A {
        A::from_view(self)
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

    /// Creates a `View` that clones its elements.
    ///
    /// This is useful when you have a `View` that computes `&T`, but you need
    /// a `View` that computes `T`.
    fn cloned<'u, U: 'u + Clone>(self) -> Cloned<Self> where
        Self: View<T=&'u U>,
    {
        Cloned(self)
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
    /// It maps `(i, j)` to `T::default()` if `i != t`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<_, _> = usize::all(3).map(|x| x + 10).diagonal().collect();
    /// assert_eq!(a.as_ref(), [
    ///     10, 0, 0,
    ///     0, 11, 0,
    ///     0, 0, 12,
    /// ]);
    /// ```
    fn diagonal(self) -> Diagonal<Self> where
        Self::T: Default,
    {
        Diagonal(self)
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
    fn map<U, F>(self, f: F) -> Map<Self, F> where
        F: Fn(Self::T) -> U,
    {
        Map(self, f)
    }

    /// Creates a `View` that applies `f` to the elements of `Self` to obtain
    /// `View`s of size `size` which are then concatenated. More precisely,
    /// `self.flat_map(f).at((i, j))` returns `f(self.at(i)).at(j)`.
    fn flat_map<V: View, F>(self, f: F, size: <V::I as Index>::Size) -> FlatMap<Self, V, F> where
        F: Fn(Self::T) -> V,
    {
        FlatMap(self, f, size)
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
        (I, V::T, J): Flatten,
        (I::Size, <V::T as Index>::Size, J::Size): Flatten,
        (I, V::I, J): Flatten,
        (I::Size, <V::I as Index>::Size, J::Size): Flatten,
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
    fn size(&self) -> <Self::I as Index>::Size { V::size(self) }
    fn at(&self, index: Self::I) -> Self::T { V::at(self, index) }
}

// ----------------------------------------------------------------------------

/// The return type of [`View::cloned()`].
#[derive(Debug, Copy, Clone)]
pub struct Cloned<V>(V);

impl<'t, T: 't + Clone, V: View<T=&'t T>> View for Cloned<V> {
    type I = V::I;
    type T = T;
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    fn at(&self, index: Self::I) -> Self::T { self.0.at(index).clone() }
}

impl_ops_for_view!(Cloned<V>);

// ----------------------------------------------------------------------------

/// The return type of [`View::enumerate()`].
#[derive(Debug, Copy, Clone)]
pub struct Enumerate<V>(V);

impl<V: View> View for Enumerate<V> {
    type I = V::I;
    type T = (V::I, V::T);
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    fn at(&self, index: Self::I) -> Self::T { (index, self.0.at(index)) }
}

impl_ops_for_view!(Enumerate<V>);

// ----------------------------------------------------------------------------

/// The return type of [`View::diagonal()`].
#[derive(Debug, Copy, Clone)]
pub struct Diagonal<V>(V);

impl<V: View> View for Diagonal<V> where
    V::I: PartialEq,
    V::T: Default,
    (V::I, V::I): Flatten,
    (<V::I as Index>::Size, <V::I as Index>::Size): Flatten,
{
    type I = (V::I, V::I);
    type T = V::T;
    fn size(&self) -> <Self::I as Index>::Size { (self.0.size(), self.0.size()) }
    fn at(&self, index: Self::I) -> Self::T {
        if index.0 == index.1 { self.0.at(index.0) } else { Default::default() }
    }
}

impl_ops_for_view!(Diagonal<V>);

// ----------------------------------------------------------------------------

/// The return type of [`View::map()`].
#[derive(Debug, Copy, Clone)]
pub struct Map<V, F>(V, F);

impl<V: View, U, F: Fn(V::T) -> U> View for Map<V, F> {
    type I = V::I;
    type T = U;
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    fn at(&self, index: Self::I) -> Self::T { self.1(self.0.at(index)) }
}

impl_ops_for_view!(Map<V, F>);

// ----------------------------------------------------------------------------

/// The return type of [`View::flat_map()`].
#[derive(Debug, Copy, Clone)]
pub struct FlatMap<V, W: View, F>(V, F, <W::I as Index>::Size);

impl<V: View, W: View, F: Fn(V::T) -> W> View for FlatMap<V, W, F> where
    (V::I, W::I): Flatten,
    (<V::I as Index>::Size, <W::I as Index>::Size): Flatten,
{
    type I = (V::I, W::I);
    type T = W::T;
    fn size(&self) -> <Self::I as Index>::Size { (self.0.size(), self.2) }
    fn at(&self, index: Self::I) -> Self::T { self.1(self.0.at(index.0)).at(index.1) }
}

impl_ops_for_view!(FlatMap<V, W: View, F>);

// ----------------------------------------------------------------------------

/// The return type of [`View::compose()`].
#[derive(Debug, Copy, Clone)]
pub struct Compose<V, W>(V, W);

impl<V: View, W: View<I=V::T>> View for Compose<V, W> {
    type I = V::I;
    type T = W::T;
    fn size(&self) -> <Self::I as Index>::Size { self.0.size() }
    fn at(&self, index: Self::I) -> Self::T { self.1.at(self.0.at(index)) }
}

impl_ops_for_view!(Compose<V, W>);

// ----------------------------------------------------------------------------

/// The return type of [`View::map_axis()`].
#[derive(Debug, Copy, Clone)]
pub struct MapAxis<V, I, W, J>(V, PhantomData<I>, W, PhantomData<J>);

impl<V: View, I: Index, W: View, J: Index> View for MapAxis<V, I, W, J> where
    W::T: Index,
    (I, W::T, J): Flatten,
    (I::Size, <W::T as Index>::Size, J::Size): Flatten,
    (I, W::I, J): Flatten,
    (I::Size, <W::I as Index>::Size, J::Size): Flatten,
    V::I: Isomorphic<(I, W::T, J)>,
    <V::I as Index>::Size: Isomorphic<(I::Size, <W::T as Index>::Size, J::Size)>,
{
    type I = (I, W::I, J);
    type T = V::T;

    fn size(&self) -> <Self::I as Index>::Size {
        let (i_size, _, j_size) = self.0.size().to_iso();
        (i_size, self.2.size(), j_size)
    }

    fn at(&self, index: Self::I) -> Self::T {
        let (i, x, j) = index;
        self.0.at(Isomorphic::from_iso((i, self.2.at(x), j)))
    }
}

impl_ops_for_view!(MapAxis<V, I, W, J>);

// ----------------------------------------------------------------------------

/// The return type of [`View::zip()`] and [`View::binary()`].
#[derive(Debug, Copy, Clone)]
pub struct Zip<V, W, B>(V, W, PhantomData<B>);

impl<V: View, W: View, B> View for Zip<V, W, B> where
    V::I: Broadcast<W::I>,
    B: Binary<V::T, W::T>,
{
    type I = <V::I as Broadcast<W::I>>::Result;
    type T = <B as Binary<V::T, W::T>>::Output;

    fn size(&self) -> <Self::I as Index>::Size {
        <V::I as Broadcast<W::I>>::size(self.0.size(), self.1.size())
    }

    fn at(&self, index: Self::I) -> Self::T {
        let (v_index, w_index) = <V::I as Broadcast<W::I>>::index(index);
        B::call(self.0.at(v_index), self.1.at(w_index))
    }
}

impl_ops_for_view!(Zip<V, W, B>);

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
    fn size(&self) -> J::Size { Isomorphic::from_iso(self.0.size()) }
    fn at(&self, index: J) -> Self::T { self.0.at(index.to_iso()) }
}

impl_ops_for_view!(Iso<V, J: Index>);

// ----------------------------------------------------------------------------

/// The return type of [`View::transpose()`].
#[derive(Debug, Copy, Clone)]
pub struct Transpose<V, I, X, Y, J>(V, PhantomData<(I, (X, Y), J)>);

impl<V: View, I: Index, X: Index, Y: Index, J: Index> View for Transpose<V, I, X, Y, J> where
    (I, (Y, X), J): Isomorphic<V::I>,
    (I::Size, (Y::Size, X::Size), J::Size): Isomorphic<<V::I as Index>::Size>,
    (X, Y): Flatten,
    (I, (X, Y), J): Flatten,
    (X::Size, Y::Size): Flatten,
    (I::Size, (X::Size, Y::Size), J::Size): Flatten,
{
    type I = (I, (X, Y), J);
    type T = V::T;

    fn size(&self) -> (I::Size, (X::Size, Y::Size), J::Size) {
        let (i, (y, x), j) = Isomorphic::from_iso(self.0.size());
        (i, (x, y), j)
    }

    fn at(&self, index: (I, (X, Y), J)) -> Self::T {
        let (i, (x, y), j) = index;
        self.0.at((i, (y, x), j).to_iso())
    }
}

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
    fn size(&self) -> J::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).1 }
    fn at(&self, index: J) -> Self::T { self.0.at((self.1, index).to_iso()) }
}

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
    fn size(&self) -> I::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).0 }
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
    fn size(&self) -> I::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).0 }
    fn at(&self, index: I) -> Self::T { self.0.at((index, self.2).to_iso()) }
}

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
    fn size(&self) -> J::Size { <(I::Size, J::Size)>::from_iso(self.0.size()).1 }
    fn at(&self, index: J) -> Self::T { self.0.column(index) }
}

impl_ops_for_view!(Columns<V, I, J>);

// ----------------------------------------------------------------------------

/// A 0-dimensional [`View`].
#[derive(Debug, Copy, Clone)]
pub struct Scalar<T: Clone>(pub T);

impl<T: Clone> View for Scalar<T> {
    type I = ();
    type T = T;
    fn size(&self) -> () { () }
    fn at(&self, _: ()) -> T { self.0.clone() }
}

impl_ops_for_view!(Scalar<T: Clone>);

// ----------------------------------------------------------------------------

/// Construct a collection from a `View`.
pub trait FromView<I: Index, T> {
    fn from_view<V: View<I=I, T=T>>(v: &V) -> Self;
}

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
pub fn fn_view<I, T, F>(size: impl Isomorphic<I::Size>, f: F) -> FnView<I, F> where
    I: Index,
    F: Fn(I) -> T,
{
    I::all(size).map(f)
}

/// The return type of [`fn_view()`].
pub type FnView<I, F> = Map<super::All<I>, F>;
