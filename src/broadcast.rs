use super::{Index, NonTuple};

/// Implemented by `Self` if type `()` can be expanded to type `Self`.
trait Expand: Index {}

impl<T: Index + NonTuple> Expand for T {}
impl<A: Index> Expand for (A,) {}
impl<A: Index, B: Index> Expand for (A, B) {}
impl<A: Index, B: Index, C: Index> Expand for (A, B, C) {}

// ----------------------------------------------------------------------------

/// `Self` implements `Broadcast<Other>` to say what happens when you zip a
/// [`View`] indexed by `Self` with one indexed by `Other`.
///
/// Roughly speaking, each axis of `self` must be the same type and size as the
/// corresponding axis of `other`, or one of them must be `()`. In the latter
/// case, the sole array element of the smaller `View` will be replicated to
/// fill out the size of the larger `View`. This is called "broadcasting".
///
/// [`View`]: super::View
pub trait Broadcast<Other: Index>: Index {
    /// The Resulting `Index` type.
    type Result: Index;

    /// The size of `Self::Result`, given the sizes of `Self` and `Other`.
    fn size(self_size: Self::Size, other_size: Other::Size) -> <Self::Result as Index>::Size;

    /// Where each `Self::Result` maps from in `Self` and in `Other`.
    fn index(index: Self::Result) -> (Self, Other);
}

impl<I: Index + NonTuple> Broadcast<I> for I {
    type Result = I;

    #[inline(always)]
    fn size(self_size: <I as Index>::Size, other_size: <I as Index>::Size) -> <I as Index>::Size {
        if self_size != other_size { panic!("Unequal sizes"); }
        self_size
    }

    #[inline(always)]
    fn index(index: I) -> (I, I) { (index, index) }
}

impl<J: Expand> Broadcast<J> for () {
    type Result = J;
    #[inline(always)]
    fn size(_: (), other_size: J::Size) -> J::Size { other_size }
    #[inline(always)]
    fn index(index: J) -> ((), J) { ((), index) }
}

impl<I: Expand> Broadcast<()> for I {
    type Result = I;
    #[inline(always)]
    fn size(self_size: I::Size, _: ()) -> I::Size { self_size }
    #[inline(always)]
    fn index(index: I) -> (I, ()) { (index, ()) }
}

impl<
    IA: Index, JA: Index,
> Broadcast<(JA,)> for (IA,) where
    IA: Broadcast<JA>,
{
    type Result = (
        <IA as Broadcast<JA>>::Result,
    );

    #[inline(always)]
    fn size(
        i_size: (IA::Size,),
        j_size: (JA::Size,),
    ) -> <Self::Result as Index>::Size {
        (
            <IA as Broadcast<JA>>::size(i_size.0, j_size.0),
        )
    }

    #[inline(always)]
    fn index(index: Self::Result) -> ((IA,), (JA,)) {
        let (ia_index, ja_index) = <IA as Broadcast<JA>>::index(index.0);
        (
            (ia_index,),
            (ja_index,),
        )
    }
}

impl<
    IA: Index, JA: Index,
    IB: Index, JB: Index,
> Broadcast<(JA, JB)> for (IA, IB) where
    IA: Broadcast<JA>,
    IB: Broadcast<JB>,
{
    type Result = (
        <IA as Broadcast<JA>>::Result,
        <IB as Broadcast<JB>>::Result,
    );

    #[inline(always)]
    fn size(
        i_size: (IA::Size, IB::Size),
        j_size: (JA::Size, JB::Size),
    ) -> <Self::Result as Index>::Size {
        (
            <IA as Broadcast<JA>>::size(i_size.0, j_size.0),
            <IB as Broadcast<JB>>::size(i_size.1, j_size.1),
        )
    }

    #[inline(always)]
    fn index(index: Self::Result) -> ((IA, IB), (JA, JB)) {
        let (ia_index, ja_index) = <IA as Broadcast<JA>>::index(index.0);
        let (ib_index, jb_index) = <IB as Broadcast<JB>>::index(index.1);
        (
            (ia_index, ib_index),
            (ja_index, jb_index),
        )
    }
}

impl<
    IA: Index, JA: Index,
    IB: Index, JB: Index,
    IC: Index, JC: Index,
> Broadcast<(JA, JB, JC)> for (IA, IB, IC) where
    IA: Broadcast<JA>,
    IB: Broadcast<JB>,
    IC: Broadcast<JC>,
{
    type Result = (
        <IA as Broadcast<JA>>::Result,
        <IB as Broadcast<JB>>::Result,
        <IC as Broadcast<JC>>::Result,
    );

    #[inline(always)]
    fn size(
        i_size: (IA::Size, IB::Size, IC::Size),
        j_size: (JA::Size, JB::Size, JC::Size),
    ) -> <Self::Result as Index>::Size {
        (
            <IA as Broadcast<JA>>::size(i_size.0, j_size.0),
            <IB as Broadcast<JB>>::size(i_size.1, j_size.1),
            <IC as Broadcast<JC>>::size(i_size.2, j_size.2),
        )
    }

    #[inline(always)]
    fn index(index: Self::Result) -> ((IA, IB, IC), (JA, JB, JC)) {
        let (ia_index, ja_index) = <IA as Broadcast<JA>>::index(index.0);
        let (ib_index, jb_index) = <IB as Broadcast<JB>>::index(index.1);
        let (ic_index, jc_index) = <IC as Broadcast<JC>>::index(index.2);
        (
            (ia_index, ib_index, ic_index),
            (ja_index, jb_index, jc_index),
        )
    }
}
