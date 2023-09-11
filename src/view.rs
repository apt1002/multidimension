use std::marker::{PhantomData};
use std::ops::{Deref};

use super::{Index, Isomorphic, Flatten, Broadcast};

/// Implemented by types that behave like an array of `Self::T`s indexed by
/// `Self::I`, but whose array elements are computed on demand.
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

    /// Creates a `View` that clones its elements.
    ///
    /// This is useful when you have a `View` that computes `&T`, but you need
    /// a `View` that computes `T`.
    fn cloned<'u, U>(self) -> Cloned<Self> where
        U: 'u + Clone,
        Self: View<T=&'u U>,
    {
        Cloned(self)
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
        Self::I: PartialEq,
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

    /// Creates a `View` that uses `self` to select elements of `other`.
    ///
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<bool, usize> = Array::new((), [2, 1]);
    /// let b: Array<usize, &str> = Array::new(3, ["apple", "body", "crane"]);
    /// let ab: Array<bool, &str> = a.compose(b).collect();
    /// assert_eq!(ab.as_ref(), ["crane", "body"])
    /// ```
    fn compose<V>(self, other: V) -> Compose<Self, V> where
        V: View<I=Self::T>,
    {
        Compose(self, other)
    }

    /// Creates a `View` that computes pairs of an element of `self` and an
    /// element of `other`.
    ///
    /// Two `View`s can be zipped only if they have compatible indices. Roughly
    /// speaking, each axis of `self` must be the same type and size as the
    /// corresponding axis of `other`, or one of them must be `()`. In the
    /// latter case, the sole array element of the smaller `View` will be
    /// replicated to fill out the size of the larger `View`. This is called
    /// "broadcasting".
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
    /// Here's an example of zipping a 1D [`Array`] with a scalar:
    /// ```
    /// use multidimension::{Index, View, Array};
    /// let a: Array<usize, usize> = usize::all(3).collect();
    /// let b: Array<(), &str> = Array::new((), ["repeated"]);
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
    fn zip<V>(self, other: V) -> Zip<Self, V> where
        V: View,
        Self::I: Broadcast<V::I>,
    {
        Zip(self, other)
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
        (J, I): Flatten,
        (J::Size, I::Size): Flatten,
    {
        self.transpose::<(), J, I, ()>().row(j)
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

// ----------------------------------------------------------------------------

/// The return type of [`View::compose()`].
#[derive(Debug, Copy, Clone)]
pub struct Zip<V, W>(V, W);

impl<V: View, W: View> View for Zip<V, W> where
    V::I: Broadcast<W::I>,
{
    type I = <V::I as Broadcast<W::I>>::Result;
    type T = (V::T, W::T);

    fn size(&self) -> <Self::I as Index>::Size {
        <V::I as Broadcast<W::I>>::size(self.0.size(), self.1.size())
    }

    fn at(&self, index: Self::I) -> Self::T {
        let (v_index, w_index) = <V::I as Broadcast<W::I>>::index(index);
        (self.0.at(v_index), self.1.at(w_index))
    }
}

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

// ----------------------------------------------------------------------------

/// The return type of [`View::column()`]
type Column<V, I, J> = Row<Transpose<V, (), J, I, ()>, J, I>;

// ----------------------------------------------------------------------------

/// A 0-dimensional [`View`].
#[derive(Debug, Copy, Clone)]
pub struct Scalar<T: Clone>(T);

impl<T: Clone> View for Scalar<T> {
    type I = ();
    type T = T;
    fn size(&self) -> () { () }
    fn at(&self, _: ()) -> T { self.0.clone() }
}

// ----------------------------------------------------------------------------

/// Construct a collection from a `View`.
pub trait FromView<I: Index, T> {
    fn from_view<V: View<I=I, T=T>>(v: &V) -> Self;
}
