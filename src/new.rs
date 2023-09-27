use super::{Index, View, FromView};

/// A buffer that accumulates items of type `T`.
pub trait Push<T> {
    /// Append `t` to `self`.
    fn push(&mut self, t: T);
}

// ----------------------------------------------------------------------------

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

impl<A: NewView> FromView<A::I, A::T> for A {
    fn from_view<V: View<I=A::I, T=A::T>>(v: &V) -> Self {
        Self::new_view(v.size(), |buffer| { v.each(|t| buffer.push(t)); })
    }
}
