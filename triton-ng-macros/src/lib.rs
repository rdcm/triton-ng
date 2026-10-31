use proc_macro::TokenStream;
use quote::quote;
use syn::{Expr, ItemImpl, parse_macro_input, visit_mut::VisitMut};

/// Replaces the first `&mut _` in the expression with `&mut __triton_out`.
/// Returns whether a replacement was made.
struct OutputSlotReplacer {
    found: bool,
}

impl VisitMut for OutputSlotReplacer {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Reference(r) = expr
            && r.mutability.is_some()
            && matches!(*r.expr, Expr::Infer(_))
        {
            self.found = true;
            *expr = syn::parse_quote!(&mut __triton_out);
            return;
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

/// Registers an `impl Backend for T` block as the Triton backend of the crate.
///
/// Keeps the impl as it is and exports the `TRITONBACKEND_*` entry points
/// Triton looks up in the backend library, each one calling the matching
/// method of `T`. A crate has one backend, so the attribute is used once per
/// `cdylib`.
///
/// # Examples
///
/// ```ignore
/// struct MyBackend;
///
/// #[triton_ng::backend]
/// impl Backend for MyBackend {
///     async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
///         // ...
///         Ok(())
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn backend(args: TokenStream, input: TokenStream) -> TokenStream {
    if !args.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[backend] takes no arguments",
        )
        .to_compile_error()
        .into();
    }

    let item = parse_macro_input!(input as ItemImpl);
    if item.trait_.is_none() {
        return syn::Error::new_spanned(
            &item.self_ty,
            "#[backend] must be placed on `impl Backend for T`",
        )
        .to_compile_error()
        .into();
    }
    let ty = &item.self_ty;

    quote! {
        #item

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_Initialize(
            backend: *mut ::std::ffi::c_void,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe { ::triton_ng::__macro_support::initialize::<#ty>(backend) }
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_Finalize(
            backend: *mut ::std::ffi::c_void,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe { ::triton_ng::__macro_support::finalize::<#ty>(backend) }
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInitialize(
            model: *mut ::std::ffi::c_void,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe { ::triton_ng::__macro_support::model_initialize::<#ty>(model) }
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelFinalize(
            model: *mut ::std::ffi::c_void,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe { ::triton_ng::__macro_support::model_finalize::<#ty>(model) }
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceInitialize(
            instance: *mut ::std::ffi::c_void,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe { ::triton_ng::__macro_support::model_instance_initialize::<#ty>(instance) }
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceFinalize(
            instance: *mut ::std::ffi::c_void,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe { ::triton_ng::__macro_support::model_instance_finalize::<#ty>(instance) }
        }

        #[unsafe(no_mangle)]
        extern "C" fn TRITONBACKEND_ModelInstanceExecute(
            instance: *mut ::std::ffi::c_void,
            requests: *const *mut ::std::ffi::c_void,
            request_count: u32,
        ) -> ::triton_ng::__macro_support::ErrorPtr {
            unsafe {
                ::triton_ng::__macro_support::model_instance_execute::<#ty>(
                    instance,
                    requests,
                    request_count,
                )
            }
        }
    }
    .into()
}
