use super::{Isomorphic, Index, impl_ops_for_view, View};

/// A dense array of `T`s indexed by `I`.
#[derive(Debug, Clone)]
pub struct Array<I: Index, T> {
    size: I::Size,
    items: Box<[T]>,
}

impl<I: Index, T> Array<I, T> {
    fn new_inner(size: I::Size, items: Box<[T]>) -> Self {
        assert_eq!(I::length(size), items.len());
        Self {size, items}
    }

    /// Constructs an `Array` of size `size` given its elements.
    ///
    /// ```
    /// use multidimension::{Index, Array};
    /// let a: Array<(usize, bool), f32> = Array::new(3, [0.0, 1.0, -1.0, 2.0, 3.0, -2.0]);
    /// assert_eq!(a[(0, false)], 0.0);
    /// assert_eq!(a[(0, true)], 1.0);
    /// assert_eq!(a[(1, false)], -1.0);
    /// assert_eq!(a[(1, true)], 2.0);
    /// assert_eq!(a[(2, false)], 3.0);
    /// assert_eq!(a[(2, true)], -2.0);
    /// ```
    pub fn new(size: impl Isomorphic<I::Size>, items: impl Into<Box<[T]>>) -> Self {
        Self::new_inner(size.to_iso(), items.into())
    }

    /// Construct an `Array` of size `size` from a function.
    ///
    /// Consider also [`fn_view()`].
    ///
    /// [`fn_view()`]: super::fn_view
    ///
    /// ```
    /// use multidimension::{Index, Array};
    /// let a: Array<usize, _> = Array::from_fn(10, |x| x % 3 == 0);
    /// assert_eq!(a.as_ref(), [true, false, false, true, false, false, true, false, false, true]);
    /// ```
    pub fn from_fn(
        size: impl Isomorphic<I::Size>,
        f: impl Fn(I) -> T,
    ) -> Self {
        super::fn_view(size, f).collect()
    }

    /// Returns the raw array elements.
    pub fn to_raw(self) -> Box<[T]> { self.items }

    /// Change the index type of this array without moving any of the items.
    pub fn iso<J: Index>(self) -> Array<J, T> where
        J: Isomorphic<I>,
        J::Size: Isomorphic<<I as Index>::Size>,
    {
        Array {size: J::Size::from_iso(self.size), items: self.items}
    }

    /// Returns a `View` that borrows the elements of `Self`.
    pub fn view(&self) -> ArrayView<I, T> { ArrayView(self) }
}

impl<I: Index, T> std::convert::AsRef<[T]> for Array<I, T> {
    fn as_ref(&self) -> &[T] { &self.items }
}

impl<I: Index, T> std::convert::AsMut<[T]> for Array<I, T> {
    fn as_mut(&mut self) -> &mut [T] { &mut self.items }
}

impl<I: Index, T> std::ops::Index<I> for Array<I, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.items[index.to_usize(self.size)]
    }
}

impl<I: Index, T> std::ops::IndexMut<I> for Array<I, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.items[index.to_usize(self.size)]
    }
}

impl<I: Index, T: Clone> View for Array<I, T> {
    type I = I;
    type T = T;
    fn size(&self) -> I::Size { self.size }
    fn len(&self) -> usize { self.as_ref().len() }
    fn at(&self, index: I) -> T { self[index].clone() }
}

impl_ops_for_view!(Array<I: Index, T>);

impl<I: Index, T> super::FromView<I, T> for Array<I, T> {
    fn from_view<V: View<I=I, T=T>>(v: &V) -> Self {
        let mut items = Vec::with_capacity(I::length(v.size()));
        v.each(|t| items.push(t));
        Self::new_inner(v.size(), items.into())
    }
}

// ----------------------------------------------------------------------------

pub struct ArrayView<'a, I: Index, T>(&'a Array<I, T>);

impl<'a, I: Index, T: Clone> View for ArrayView<'a, I, T> {
    type I = I;
    type T = &'a T;
    fn size(&self) -> I::Size { self.0.size() }
    fn at(&self, index: I) -> &'a T { &self.0[index] }
}
