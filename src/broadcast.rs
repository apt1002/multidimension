use super::{ArrayIndex, NonTuple, Flatten};

/// Implemented by `Self` if type `()` can be expanded to type `Self`.
pub trait Expand: ArrayIndex {}

impl<T: ArrayIndex + NonTuple> Expand for T {}

impl<A: ArrayIndex> Expand for (A,) where
    (A,): Flatten,
    (A::Size,): Flatten,
{}

impl<A: ArrayIndex, B: ArrayIndex> Expand for (A, B) where
    (A, B): Flatten,
    (A::Size, B::Size): Flatten,
{}

impl<A: ArrayIndex, B: ArrayIndex, C: ArrayIndex> Expand for (A, B, C) where
    (A, B, C): Flatten,
    (A::Size, B::Size, C::Size): Flatten,
{}

// ----------------------------------------------------------------------------

/// `Self` implements `Broadcast<Other>` to say what happens when you zip a
/// `View` indexed by `Self` with one indexed by `Other`.
pub trait Broadcast<Other: ArrayIndex>: ArrayIndex {
    /// The Resulting `ArrayIndex` type.
    type Result: ArrayIndex;

    /// The size of `Self::Result`, given the sizes of `Self` and `Other`.
    fn size(self_size: Self::Size, other_size: Other::Size) -> <Self::Result as ArrayIndex>::Size;

    /// Where each `Self::Result` maps from in `Self` and in `Other`.
    fn index(index: Self::Result) -> (Self, Other);
}

impl<I: ArrayIndex + NonTuple> Broadcast<I> for I {
    type Result = I;

    fn size(self_size: <I as ArrayIndex>::Size, other_size: <I as ArrayIndex>::Size) -> <I as ArrayIndex>::Size {
        if self_size != other_size { panic!("Unequal sizes"); }
        self_size
    }

    fn index(index: I) -> (I, I) { (index, index) }
}

impl<J: Expand> Broadcast<J> for () {
    type Result = J;
    fn size(_: (), other_size: J::Size) -> J::Size { other_size }
    fn index(index: J) -> ((), J) { ((), index) }
}

impl<I: Expand> Broadcast<()> for I {
    type Result = I;
    fn size(self_size: I::Size, _: ()) -> I::Size { self_size }
    fn index(index: I) -> (I, ()) { (index, ()) }
}

impl<
    IA: ArrayIndex, JA: ArrayIndex,
> Broadcast<(JA,)> for (IA,) where
    IA: Broadcast<JA>,
{
    type Result = (
        <IA as Broadcast<JA>>::Result,
    );

    fn size(
        i_size: (IA::Size,),
        j_size: (JA::Size,),
    ) -> <Self::Result as ArrayIndex>::Size {
        (
            <IA as Broadcast<JA>>::size(i_size.0, j_size.0),
        )
    }

    fn index(index: Self::Result) -> ((IA,), (JA,)) {
        let (ia_index, ja_index) = <IA as Broadcast<JA>>::index(index.0);
        (
            (ia_index,),
            (ja_index,),
        )
    }
}

impl<
    IA: ArrayIndex, JA: ArrayIndex,
    IB: ArrayIndex, JB: ArrayIndex,
> Broadcast<(JA, JB)> for (IA, IB) where
    IA: Broadcast<JA>,
    IB: Broadcast<JB>,
    (IA, IB): Flatten,
    (IA::Size, IB::Size): Flatten,
    (JA, JB): Flatten,
    (JA::Size, JB::Size): Flatten,
    (
        <IA as Broadcast<JA>>::Result,
        <IB as Broadcast<JB>>::Result,
    ): Flatten,
    (
        <<IA as Broadcast<JA>>::Result as ArrayIndex>::Size,
        <<IB as Broadcast<JB>>::Result as ArrayIndex>::Size,
    ): Flatten,
{
    type Result = (
        <IA as Broadcast<JA>>::Result,
        <IB as Broadcast<JB>>::Result,
    );

    fn size(
        i_size: (IA::Size, IB::Size),
        j_size: (JA::Size, JB::Size),
    ) -> <Self::Result as ArrayIndex>::Size {
        (
            <IA as Broadcast<JA>>::size(i_size.0, j_size.0),
            <IB as Broadcast<JB>>::size(i_size.1, j_size.1),
        )
    }

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
    IA: ArrayIndex, JA: ArrayIndex,
    IB: ArrayIndex, JB: ArrayIndex,
    IC: ArrayIndex, JC: ArrayIndex,
> Broadcast<(JA, JB, JC)> for (IA, IB, IC) where
    IA: Broadcast<JA>,
    IB: Broadcast<JB>,
    IC: Broadcast<JC>,
    (IA, IB, IC): Flatten,
    (IA::Size, IB::Size, IC::Size): Flatten,
    (JA, JB, JC): Flatten,
    (JA::Size, JB::Size, JC::Size): Flatten,
    (
        <IA as Broadcast<JA>>::Result,
        <IB as Broadcast<JB>>::Result,
        <IC as Broadcast<JC>>::Result,
    ): Flatten,
    (
        <<IA as Broadcast<JA>>::Result as ArrayIndex>::Size,
        <<IB as Broadcast<JB>>::Result as ArrayIndex>::Size,
        <<IC as Broadcast<JC>>::Result as ArrayIndex>::Size,
    ): Flatten,
{
    type Result = (
        <IA as Broadcast<JA>>::Result,
        <IB as Broadcast<JB>>::Result,
        <IC as Broadcast<JC>>::Result,
    );

    fn size(
        i_size: (IA::Size, IB::Size, IC::Size),
        j_size: (JA::Size, JB::Size, JC::Size),
    ) -> <Self::Result as ArrayIndex>::Size {
        (
            <IA as Broadcast<JA>>::size(i_size.0, j_size.0),
            <IB as Broadcast<JB>>::size(i_size.1, j_size.1),
            <IC as Broadcast<JC>>::size(i_size.2, j_size.2),
        )
    }

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
