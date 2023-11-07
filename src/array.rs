use super::{Isomorphic, Size, Index, impl_ops_for_view, View, ViewRef, ViewMut, impl_ops_for_memoryview};

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
        mut f: impl FnMut(I) -> T,
    ) -> Self {
        let size = size.to_iso();
        let mut items = Vec::with_capacity(I::length(size));
        size.each(|i| items.push(f(i)));
        Self::new_inner(size, items.into())
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
}

impl<I: Index, T> std::convert::AsRef<[T]> for Array<I, T> {
    fn as_ref(&self) -> &[T] { &self.items }
}

impl<I: Index, T> std::convert::AsMut<[T]> for Array<I, T> {
    fn as_mut(&mut self) -> &mut [T] { &mut self.items }
}

impl<I: Index, T: Clone> View for Array<I, T> {
    type I = I;
    type T = T;
    #[inline(always)]
    fn size(&self) -> I::Size { self.size }
    #[inline(always)]
    fn len(&self) -> usize { self.as_ref().len() }
    #[inline(always)]
    fn at(&self, index: I) -> T { self[index].clone() }
}

impl<I: Index, T: Clone> ViewRef for Array<I, T> {
    #[inline(always)]
    fn at_ref(&self, index: Self::I) -> &Self::T { &self.items[index.to_usize(self.size)] }
}

impl<I: Index, T: Clone> ViewMut for Array<I, T> {
    #[inline(always)]
    fn at_mut(&mut self, index: Self::I) -> &mut Self::T { &mut self.items[index.to_usize(self.size)] }
}

impl_ops_for_view!(Array<I: Index, T>);
impl_ops_for_memoryview!(Array<I: Index, T>);

// ----------------------------------------------------------------------------

impl<T> super::Push<T> for Vec<T> {
    fn push(&mut self, t: T) { Vec::push(self, t); }
}

impl<I: Index, T: Clone> super::NewView for Array<I, T> {
    type Buffer = Vec<T>;

    fn new_view(
        size: I::Size,
        callback: impl FnOnce(&mut Self::Buffer),
    ) -> Self {
        let mut buffer = Vec::with_capacity(I::length(size));
        callback(&mut buffer);
        Self::new_inner(size, buffer.into())
    }
}
