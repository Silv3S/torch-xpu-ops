// Minimal reproducer: the SAME float logspace kernel gives a different result
// depending only on whether sibling dtype kernels share the translation unit.
//
// Build the float-only variant:   -DEXTRA_DTYPES=0
// Build the all-dtypes variant:    -DEXTRA_DTYPES=1
// The float kernel source is byte-identical between the two; the only change is
// the presence of the other-dtype LogspaceFunctor instantiations in the module.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <complex>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>

using bf16 = sycl::ext::oneapi::bfloat16;
using half = sycl::half;

static uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

template <typename T> struct OpMath { using type = T; };
template <> struct OpMath<half> { using type = float; };
template <> struct OpMath<bf16> { using type = float; };
template <typename T> using opmath_t = typename OpMath<T>::type;

// Faithful LogspaceFunctor<scalar_t, step_type> (real, non-complex branch).
template <typename scalar_t, typename step_type>
struct LogspaceFunctor {
    scalar_t base_;
    scalar_t start_;
    step_type step_;
    scalar_t operator()(int64_t ind) const {
        using op = opmath_t<step_type>;
        return static_cast<scalar_t>(sycl::pow(
            static_cast<op>(base_),
            static_cast<op>(static_cast<op>(start_) + step_ * ind)));
    }
};

template <typename scalar_t, typename step_type>
struct LogspaceFunctorComplex {
    scalar_t base_;
    scalar_t start_;
    step_type step_;
    scalar_t operator()(int64_t ind) const {
        return std::pow(base_, start_ + step_ * static_cast<step_type>(ind));
    }
};

template <typename scalar_t, typename step_type, class Tag>
void run_logspace(sycl::queue& q, scalar_t* out_host, int64_t steps,
                  scalar_t base, scalar_t start, step_type step) {
    sycl::buffer<scalar_t, 1> bOut(out_host, sycl::range<1>(steps));
    LogspaceFunctor<scalar_t, step_type> f{base, start, step};
    q.submit([&](sycl::handler& h) {
        sycl::accessor o_(bOut, h, sycl::write_only, sycl::no_init);
        h.parallel_for<Tag>(sycl::range<1>(steps), [=](sycl::id<1> i) {
            o_[i] = f(static_cast<int64_t>(i[0]));
        });
    }).wait();
}

template <typename scalar_t, typename step_type, class Tag>
void run_logspace_complex(sycl::queue& q, scalar_t* out_host, int64_t steps,
                          scalar_t base, scalar_t start, step_type step) {
    sycl::buffer<scalar_t, 1> bOut(out_host, sycl::range<1>(steps));
    LogspaceFunctorComplex<scalar_t, step_type> f{base, start, step};
    q.submit([&](sycl::handler& h) {
        sycl::accessor o_(bOut, h, sycl::write_only, sycl::no_init);
        h.parallel_for<Tag>(sycl::range<1>(steps), [=](sycl::id<1> i) {
            o_[i] = f(static_cast<int64_t>(i[0]));
        });
    }).wait();
}

struct Tf; struct Td; struct Th; struct Tbf;
struct Ti8; struct Tu8; struct Ti16; struct Ti32; struct Ti64;
struct Tcf; struct Tcd;

int main() {
    sycl::queue q;
    const int64_t steps = 2;

    // The kernel we care about: float logspace(1,0,2) -> out[0] = 10^1.
    // This block is IDENTICAL in both build variants.
    float out_f[2] = {0, 0};
    run_logspace<float, float, Tf>(q, out_f, steps, 10.0f, 1.0f, -1.0f);

#if EXTRA_DTYPES
    // The other dtype kernels torch instantiates in the SAME TU. Their mere
    // presence flips the float kernel's result via module-wide ContractionOff.
    double out_d[2] = {0, 0};
    run_logspace<double, double, Td>(q, out_d, steps, 10.0, 1.0, -1.0);

    half out_h[2] = {half(0), half(0)};
    run_logspace<half, half, Th>(q, out_h, steps, half(10.0f), half(1.0f), half(-1.0f));

    bf16 out_bf[2] = {bf16(0), bf16(0)};
    run_logspace<bf16, bf16, Tbf>(q, out_bf, steps, bf16(10.0f), bf16(1.0f), bf16(-1.0f));

    int8_t   out_i8[2]  = {0, 0};
    run_logspace<int8_t, float, Ti8>(q, out_i8, steps, (int8_t)10, (int8_t)1, -1.0f);
    uint8_t  out_u8[2]  = {0, 0};
    run_logspace<uint8_t, float, Tu8>(q, out_u8, steps, (uint8_t)10, (uint8_t)1, -1.0f);
    int16_t  out_i16[2] = {0, 0};
    run_logspace<int16_t, float, Ti16>(q, out_i16, steps, (int16_t)10, (int16_t)1, -1.0f);
    int32_t  out_i32[2] = {0, 0};
    run_logspace<int32_t, float, Ti32>(q, out_i32, steps, (int32_t)10, (int32_t)1, -1.0f);
    int64_t  out_i64[2] = {0, 0};
    run_logspace<int64_t, float, Ti64>(q, out_i64, steps, (int64_t)10, (int64_t)1, -1.0f);

    std::complex<float> out_cf[2] = {{0, 0}, {0, 0}};
    run_logspace_complex<std::complex<float>, std::complex<float>, Tcf>(
        q, out_cf, steps, {10.0f, 0.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f});
    std::complex<double> out_cd[2] = {{0, 0}, {0, 0}};
    run_logspace_complex<std::complex<double>, std::complex<double>, Tcd>(
        q, out_cd, steps, {10.0, 0.0}, {1.0, 0.0}, {-1.0, 0.0});
#endif

    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
#if EXTRA_DTYPES
    std::cout << "TU variant: FLOAT + all other dtype kernels (EXTRA_DTYPES=1)\n";
#else
    std::cout << "TU variant: FLOAT kernel only (EXTRA_DTYPES=0)\n";
#endif
    std::cout << std::setprecision(17);
    std::cout << "FLOAT logspace[0] = pow(10,1) = " << out_f[0]
              << "  [bits 0x" << std::hex << bits(out_f[0]) << std::dec << "]\n";
    std::cout << "  (exact 10.0 = 0x41200000; 1 ULP high = 0x41200001)\n";
    return 0;
}
