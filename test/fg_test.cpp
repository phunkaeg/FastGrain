/*
    fg_test.cpp - standalone CPU harness for FastGrain_Shared.h (no AE needed).

    Checks:
      1. grain-only view: mean ~ 0.5, std ~ amplitude, no NaN
      2. ROI independence: rendering a sub-rectangle reproduces the same pixels
         as the full-frame render (validates origin / lattice-base maths)
      3. softness keeps variance ~ constant (variance compensation)
      4. temporal: frameBlend produces intermediate, unit-variance noise
      5. rough single-thread timing for a 1920x1080 frame

    Build:  cl /O2 /EHsc /I..\src fg_test.cpp
*/
#include "FastGrain_Shared.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

struct Rect { int left, top, right, bottom; };

static void BuildLayout(FGParams& P, const Rect& rect, const float sizeX[4], const float sizeY[4])
{
    const float Xmin = P.originX;
    const float Xmax = P.originX + (float)(rect.right - rect.left - 1) * P.dsx;
    const float Ymin = P.originY;
    const float Ymax = P.originY + (float)(rect.bottom - rect.top - 1) * P.dsy;
    const int   r    = P.blurRadius;
    int rowOff[4], dimsW[4], dimsH[4], baseI[4], baseJ[4];
    float invSx[4], invSy[4];
    int off = 0, maxW = 0;
    for (int p = 0; p < 4; ++p) {
        invSx[p] = 1.0f / sizeX[p];
        invSy[p] = 1.0f / sizeY[p];
        int Imin = (int)floorf(Xmin * invSx[p]) - 1 - r;
        int Imax = (int)floorf(Xmax * invSx[p]) + 2 + r;
        int Jmin = (int)floorf(Ymin * invSy[p]) - 1 - r;
        int Jmax = (int)floorf(Ymax * invSy[p]) + 2 + r;
        baseI[p] = Imin; baseJ[p] = Jmin;
        dimsW[p] = Imax - Imin + 1; dimsH[p] = Jmax - Jmin + 1;
        rowOff[p] = off; off += dimsH[p]; maxW = std::max(maxW, dimsW[p]);
    }
    P.p0RowOff = rowOff[0]; P.p0DimsW = dimsW[0]; P.p0DimsH = dimsH[0]; P.p0BaseI = baseI[0]; P.p0BaseJ = baseJ[0]; P.p0InvSx = invSx[0]; P.p0InvSy = invSy[0];
    P.p1RowOff = rowOff[1]; P.p1DimsW = dimsW[1]; P.p1DimsH = dimsH[1]; P.p1BaseI = baseI[1]; P.p1BaseJ = baseJ[1]; P.p1InvSx = invSx[1]; P.p1InvSy = invSy[1];
    P.p2RowOff = rowOff[2]; P.p2DimsW = dimsW[2]; P.p2DimsH = dimsH[2]; P.p2BaseI = baseI[2]; P.p2BaseJ = baseJ[2]; P.p2InvSx = invSx[2]; P.p2InvSy = invSy[2];
    P.p3RowOff = rowOff[3]; P.p3DimsW = dimsW[3]; P.p3DimsH = dimsH[3]; P.p3BaseI = baseI[3]; P.p3BaseJ = baseJ[3]; P.p3InvSx = invSx[3]; P.p3InvSy = invSy[3];
    P.lattMaxW = maxW; P.lattTotalH = off; P.lattPitch = maxW;
}

static FGParams Defaults()
{
    FGParams P; std::memset(&P, 0, sizeof(P));
    P.dsx = P.dsy = 1.0f;
    P.blurRadius = 0; P.blurSigma = 1.0f; P.blurNorm = 1.0f;
    P.frameA = 0; P.frameB = 1; P.frameBlend = 0.0f; P.seed = 0;
    P.saturation = 1.0f;
    P.tintR = P.tintG = P.tintB = 1.0f;
    P.ampR = P.ampG = P.ampB = 0.05f;
    P.shR = P.shG = P.shB = 1.0f; P.miR = P.miG = P.miB = 1.0f; P.hiR = P.hiG = P.hiB = 1.0f;
    P.midpoint = 0.5f;
    P.blendMode = FG_BLEND_FILM; P.mixOriginal = 0.0f; P.viewMode = FG_VIEW_RESULT; P.clampOutput = 0;
    return P;
}

/* full pipeline on the CPU for a rect; returns RGBA float image of rect size */
static std::vector<FGPixel> Render(FGParams P, const Rect& rect, float size, float softness, const std::vector<FGPixel>* srcImg, int srcStride)
{
    P.width = rect.right - rect.left; P.height = rect.bottom - rect.top;
    P.originX = rect.left * P.dsx; P.originY = rect.top * P.dsy;
    if (softness < 0.05f) { P.blurRadius = 0; P.blurSigma = 1; P.blurNorm = 1; }
    else { P.blurSigma = softness; P.blurRadius = std::min((int)ceilf(3 * softness), FG_MAX_BLUR_RADIUS); P.blurNorm = fg_blur_norm(P.blurRadius, P.blurSigma); }
    float sx[4], sy[4];
    for (int p = 0; p < 4; ++p) { sx[p] = std::max(size, P.dsx); sy[p] = std::max(size, P.dsy); }
    BuildLayout(P, rect, sx, sy);

    size_t cells = (size_t)P.lattPitch * P.lattTotalH;
    std::vector<float> latt(cells), temp(P.blurRadius > 0 ? cells : 0);
    for (int row = 0; row < P.lattTotalH; ++row)
        for (int i = 0; i < P.lattMaxW; ++i) latt[(size_t)row * P.lattPitch + i] = fg_gen_cell(i, row, P);
    if (P.blurRadius > 0) {
        for (int row = 0; row < P.lattTotalH; ++row)
            for (int i = 0; i < P.lattMaxW; ++i) temp[(size_t)row * P.lattPitch + i] = fg_blur_h_cell(latt.data(), i, row, P);
        for (int row = 0; row < P.lattTotalH; ++row)
            for (int i = 0; i < P.lattMaxW; ++i) latt[(size_t)row * P.lattPitch + i] = fg_blur_v_cell(temp.data(), i, row, P);
    }
    std::vector<FGPixel> out((size_t)P.width * P.height);
    for (int y = 0; y < P.height; ++y)
        for (int x = 0; x < P.width; ++x) {
            FGPixel s;
            if (srcImg) s = (*srcImg)[(size_t)(rect.top + y) * srcStride + (rect.left + x)];
            else { s.r = s.g = s.b = 0.5f; s.a = 1.0f; }
            out[(size_t)y * P.width + x] = fg_composite(latt.data(), P, s, P.originX + x * P.dsx, P.originY + y * P.dsy);
        }
    return out;
}

static void Stats(const std::vector<FGPixel>& img, double& mean, double& stddev, int& nans, int ch)
{
    double s = 0, s2 = 0; nans = 0;
    for (const FGPixel& p : img) {
        float v = ch == 0 ? p.r : (ch == 1 ? p.g : p.b);
        if (std::isnan(v) || std::isinf(v)) { nans++; continue; }
        s += v; s2 += (double)v * v;
    }
    double n = (double)img.size();
    mean = s / n; stddev = sqrt(std::max(0.0, s2 / n - mean * mean));
}

int main()
{
    int fails = 0;
    const Rect full = { 0, 0, 640, 360 };

    /* 1. grain-only statistics at several sizes */
    for (float size : { 1.0f, 2.0f, 3.7f }) {
        FGParams P = Defaults(); P.viewMode = FG_VIEW_GRAIN;
        auto img = Render(P, full, size, 0.0f, nullptr, 0);
        double mean, sd; int nans;
        Stats(img, mean, sd, nans, 0);
        bool ok = nans == 0 && fabs(mean - 0.5) < 0.005 && fabs(sd - 0.05) < 0.006;
        printf("[%s] grain-only size=%.1f  mean=%.4f std=%.4f (want 0.5 / 0.05) nans=%d\n", ok ? "PASS" : "FAIL", size, mean, sd, nans);
        if (!ok) fails++;
    }

    /* 2. ROI independence */
    {
        FGParams P = Defaults(); P.viewMode = FG_VIEW_GRAIN;
        for (float size : { 1.0f, 2.5f }) for (float soft : { 0.0f, 0.8f }) {
            auto whole = Render(P, full, size, soft, nullptr, 0);
            Rect sub = { 123, 77, 400, 300 };
            auto part = Render(P, sub, size, soft, nullptr, 0);
            double maxDiff = 0;
            for (int y = 0; y < sub.bottom - sub.top; ++y)
                for (int x = 0; x < sub.right - sub.left; ++x) {
                    const FGPixel& a = whole[(size_t)(sub.top + y) * (full.right) + (sub.left + x)];
                    const FGPixel& b = part[(size_t)y * (sub.right - sub.left) + x];
                    maxDiff = std::max(maxDiff, (double)fabs(a.r - b.r));
                    maxDiff = std::max(maxDiff, (double)fabs(a.g - b.g));
                }
            bool ok = maxDiff < 1e-5;
            printf("[%s] ROI independence size=%.1f soft=%.1f  max|diff|=%.2e\n", ok ? "PASS" : "FAIL", size, soft, maxDiff);
            if (!ok) fails++;
        }
    }

    /* 3. softness keeps variance */
    for (float soft : { 0.5f, 1.5f, 3.0f }) {
        FGParams P = Defaults(); P.viewMode = FG_VIEW_GRAIN;
        auto img = Render(P, full, 1.0f, soft, nullptr, 0);
        double mean, sd; int nans; Stats(img, mean, sd, nans, 1);
        bool ok = nans == 0 && fabs(sd - 0.05) < 0.008;
        printf("[%s] softness=%.1f  std=%.4f (want ~0.05) nans=%d\n", ok ? "PASS" : "FAIL", soft, sd, nans);
        if (!ok) fails++;
    }

    /* 4. temporal blend */
    {
        FGParams P = Defaults(); P.viewMode = FG_VIEW_GRAIN; P.frameA = 3; P.frameB = 4; P.frameBlend = 0.5f;
        auto img = Render(P, full, 1.0f, 0.0f, nullptr, 0);
        P.frameBlend = 0.0f;  auto a = Render(P, full, 1.0f, 0.0f, nullptr, 0);
        P.frameA = 4; P.frameB = 5; auto b = Render(P, full, 1.0f, 0.0f, nullptr, 0);
        double mean, sd; int nans; Stats(img, mean, sd, nans, 2);
        double dA = 0, dB = 0;
        for (size_t i = 0; i < img.size(); ++i) { dA += fabs(img[i].b - a[i].b); dB += fabs(img[i].b - b[i].b); }
        dA /= img.size(); dB /= img.size();
        bool ok = nans == 0 && fabs(sd - 0.05) < 0.006 && dA > 0.01 && dB > 0.01 && fabs(dA - dB) < 0.005;
        printf("[%s] temporal blend 0.5: std=%.4f  meanDiffA=%.4f meanDiffB=%.4f\n", ok ? "PASS" : "FAIL", sd, dA, dB);
        if (!ok) fails++;
    }

    /* 5. blend modes on a grey ramp: mean preserved roughly, no NaN, film mode positive */
    {
        std::vector<FGPixel> ramp((size_t)full.right * full.bottom);
        for (int y = 0; y < full.bottom; ++y) for (int x = 0; x < full.right; ++x) {
            float v = (float)x / (full.right - 1);
            ramp[(size_t)y * full.right + x] = { v, v, v, 1.0f };
        }
        for (int mode = 0; mode <= 4; ++mode) {
            FGParams P = Defaults(); P.blendMode = mode;
            auto img = Render(P, full, 1.5f, 0.5f, &ramp, full.right);
            double mean, sd; int nans; Stats(img, mean, sd, nans, 0);
            double minv = 1e9;
            for (auto& p : img) minv = std::min(minv, (double)p.r);
            bool ok = nans == 0 && fabs(mean - 0.5) < 0.01 && (mode != FG_BLEND_FILM || minv >= 0.0);
            printf("[%s] blend mode %d on ramp: mean=%.4f std=%.4f min=%.4f nans=%d\n", ok ? "PASS" : "FAIL", mode, mean, sd, minv, nans);
            if (!ok) fails++;
        }
    }

    /* 6. timing, single thread, 1080p, size 1, softness 0.5 */
    {
        FGParams P = Defaults();
        Rect hd = { 0, 0, 1920, 1080 };
        auto t0 = std::chrono::high_resolution_clock::now();
        auto img = Render(P, hd, 1.0f, 0.5f, nullptr, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[INFO] 1920x1080 single-thread full pipeline: %.1f ms  (%.3f)\n", ms, img[12345].r);
    }

    printf("\n%s (%d failures)\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fails);
    return fails ? 1 : 0;
}
