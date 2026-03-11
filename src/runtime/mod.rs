//! Model execution and learning logic.
//! The CPU reference path in `forward`, `backward`, and `training`
//! is the main teaching path; `backend` holds optional acceleration.

pub mod backend;
pub mod backward;
pub mod chat;
pub mod eval;
pub mod forward;
pub mod profile;
pub mod sampling;
pub mod train_cache;
pub mod training;
pub mod workspace;
