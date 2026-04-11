use proc_macro::TokenStream;
use quote::quote;
use syn::{Expr, parse_macro_input, visit_mut::VisitMut};

/// Replaces the first `&mut _` in the expression with `&mut __triton_out`.
/// Returns whether a replacement was made.
struct OutputSlotReplacer {
    found: bool,
}

impl VisitMut for OutputSlotReplacer {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Reference(r) = expr {
            if r.mutability.is_some() && matches!(*r.expr, Expr::Infer(_)) {
                self.found = true;
                *expr = syn::parse_quote!(&mut __triton_out);
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }
}

/// Calls a Triton FFI function and converts the `*mut TRITONSERVER_Error` return into a `Result`.
///
/// **Simple form** — returns `Result<(), TritonError>`:
/// ```ignore
/// triton_call!(triton_sys::SomeFn(arg1, arg2))?;
/// ```
///
/// **Output pointer form** — use `&mut _` to mark the output slot.
/// The macro declares the pointer, calls the function, checks the error and null,
/// and returns `Result<*mut T, TritonError>`:
/// ```ignore
/// let ptr = triton_call!(triton_sys::SomeFn(&mut _, arg1))?;
/// ```
#[proc_macro]
pub fn triton_call(input: TokenStream) -> TokenStream {
    let mut call_expr = parse_macro_input!(input as Expr);

    let mut replacer = OutputSlotReplacer { found: false };
    replacer.visit_expr_mut(&mut call_expr);

    if replacer.found {
        quote! {{
            let mut __triton_out = ::std::ptr::null_mut();
            let __triton_err = unsafe { #call_expr };
            if !__triton_err.is_null() {
                ::std::result::Result::Err(unsafe { crate::error::TritonError::new(__triton_err) })
            } else if __triton_out.is_null() {
                ::std::result::Result::Err(crate::error::TritonError::from_message("unexpected null pointer"))
            } else {
                ::std::result::Result::Ok(__triton_out)
            }
        }}
        .into()
    } else {
        quote! {{
            let __triton_err = unsafe { #call_expr };
            if __triton_err.is_null() {
                ::std::result::Result::Ok(())
            } else {
                ::std::result::Result::Err(unsafe { crate::error::TritonError::new(__triton_err) })
            }
        }}
        .into()
    }
}
