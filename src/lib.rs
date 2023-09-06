mod index;
pub use index::{ArrayIndex, StaticIndex};

pub mod tuple;
pub use tuple::{NonTuple, Flatten, Isomorphic};

mod broadcast;
pub use broadcast::{Broadcast};

pub mod view;
pub use view::{View, FromView};

mod array;
pub use array::{Array};
