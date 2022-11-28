// [[file:../enm.onte::a8b9ab5d][a8b9ab5d]]
#![deny(warnings)]
// a8b9ab5d ends here

// [[file:../enm.onte::02256b51][02256b51]]
mod enm;
// 02256b51 ends here

// [[file:../enm.onte::*docs][docs:1]]
#[cfg(feature = "adhoc")]
/// Docs for local mods
pub mod docs {
    macro_rules! export_doc {
        ($l:ident) => {
            pub mod $l {
                pub use crate::$l::*;
            }
        };
    }

    // export_doc!(codec);
}
// docs:1 ends here
