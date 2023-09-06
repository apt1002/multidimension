use super::{Isomorphic, ArrayIndex, View};

/// A dense array of `T`s indexed by `I`.
#[derive(Debug, Clone)]
pub struct Array<I: ArrayIndex, T> {
    size: I::Size,
    items: Box<[T]>,
}

impl<I: ArrayIndex, T> Array<I, T> {
    /// Constructs an `Array` of size `size` given its elements.
    ///
    /// ```
    /// use multidimension::{ArrayIndex, Array};
    /// let a: Array<(usize, bool), f32> = Array::new((3, ()), [0.0, 1.0, -1.0, 2.0, 3.0, -2.0]);
    /// assert_eq!(a[(0, false)], 0.0);
    /// assert_eq!(a[(0, true)], 1.0);
    /// assert_eq!(a[(1, false)], -1.0);
    /// assert_eq!(a[(1, true)], 2.0);
    /// assert_eq!(a[(2, false)], 3.0);
    /// assert_eq!(a[(2, true)], -2.0);
    /// ```
    pub fn new(size: I::Size, items: impl Into<Box<[T]>>) -> Self {
        let items = items.into();
        assert_eq!(I::length(size), items.len());
        Self {size, items}
    }

    /// Construct an `Array` of size `size` from a function.
    ///
    /// ```
    /// use multidimension::{ArrayIndex, Array};
    /// let a: Array<usize, _> = Array::from_fn(10, |x| x % 3 == 0);
    /// assert_eq!(a.as_ref(), [true, false, false, true, false, false, true, false, false, true]);
    /// ```
    pub fn from_fn(
        size: impl Isomorphic<I::Size>,
        mut f: impl FnMut(I) -> T,
    ) -> Self {
        let size = size.to_iso();
        let mut items = Vec::with_capacity(I::length(size));
        I::each(size, |i| items.push(f(i)));
        Self {size, items: items.into()}
    }

    /// The run-time representation of the size of `Self`.
    ///
    /// Use `self.len()` to obtain the number of elements in `Self`.
    pub fn size(&self) -> I::Size { self.size }

    /// The number of elements in `Self`.
    pub fn len(&self) -> usize { self.as_ref().len() }

    /// Converts `index` to a `usize`.
    pub fn index(&self, index: I) -> usize { index.as_usize(self.size) }

    /// Change the index type of this array without moving any of the items.
    pub fn iso<J: ArrayIndex>(self) -> Array<J, T> where
        J: Isomorphic<I>,
        J::Size: Isomorphic<<I as ArrayIndex>::Size>,
    {
        Array {size: J::Size::from_iso(self.size), items: self.items}
    }

    /// Returns a `View` that borrows the elements of `Self`.
    pub fn view(&self) -> ArrayView<I, T> { ArrayView(self) }
}

impl<I: ArrayIndex, T> std::convert::AsRef<[T]> for Array<I, T> {
    fn as_ref(&self) -> &[T] { &self.items }
}

impl<I: ArrayIndex, T> std::convert::AsMut<[T]> for Array<I, T> {
    fn as_mut(&mut self) -> &mut [T] { &mut self.items }
}

impl<I: ArrayIndex, T> std::ops::Index<I> for Array<I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.items[self.index(index)]
    }
}

impl<I: ArrayIndex, T> std::ops::IndexMut<I> for Array<I, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.items[self.index(index)]
    }
}

impl<I: ArrayIndex, T: Clone> View for Array<I, T> {
    type I = I;
    type T = T;
    fn size(&self) -> I::Size { Array::size(self) }
    fn at(&self, index: I) -> T { self[index].clone() }
}

impl<I: ArrayIndex, T> super::FromView<I, T> for Array<I, T> {
    fn from_view<V: View<I=I, T=T>>(v: &V) -> Self {
        Self::from_fn(v.size(), |i| v.at(i))
    }
}

// ----------------------------------------------------------------------------

pub struct ArrayView<'a, I: ArrayIndex, T>(&'a Array<I, T>);

impl<'a, I: ArrayIndex, T: Clone> View for ArrayView<'a, I, T> {
    type I = I;
    type T = &'a T;
    fn size(&self) -> I::Size { self.0.size() }
    fn at(&self, index: I) -> &'a T { &self.0[index] }
}
