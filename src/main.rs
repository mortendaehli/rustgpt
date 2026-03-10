fn main() {
    if let Err(err) = rustgpt::run(std::env::args()) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
