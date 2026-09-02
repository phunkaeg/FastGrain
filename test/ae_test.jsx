/*
    ae_test.jsx - drive After Effects to exercise the Fast Grain plug-in.
    Run from inside After Effects: File > Scripts > Run Script File... (command-line -r launches
    produced no output on this machine). Needs "Allow Scripts to Write Files and Access Network".
    Appends to test\out\log.txt after every step (survives crashes) and writes PNG frames + benchmark
    AVIs into test\out\. Renders FastGrain on CPU and CUDA, then benchmarks against Adobe's Add Grain.
    ExtendScript (ES3): no let/const/forEach.
*/
(function () {
    /* output next to this script: FastGrain\test\out\ */
    var OUT_DIR = new File($.fileName).parent.fsName.replace(/\\/g, "/") + "/out";
    app.exitAfterLaunchAndEval = false;      /* keep AE open for inspection */

    function L(s) {
        var f = new File(OUT_DIR + "/log.txt"); f.encoding = "UTF-8";
        if (f.open("a")) { f.writeln(String(s)); f.close(); }
    }
    function png(comp, t, name) {
        var f = new File(OUT_DIR + "/" + name + ".png");
        var t0 = new Date().getTime();
        try { comp.saveFrameToPng(t, f); L("saved " + name + ".png exists=" + f.exists + " (" + (new Date().getTime() - t0) + " ms)"); }
        catch (e) { L("saveFrameToPng FAILED for " + name + ": " + e.toString()); }
    }
    function setGPU(type, label) {
        try { app.project.gpuAccelType = type; L("gpuAccelType -> " + label + " (now " + app.project.gpuAccelType + ")"); return true; }
        catch (e) { L("could not set gpuAccelType " + label + ": " + e.toString()); return false; }
    }
    function hasType(t) {
        var a = app.availableGPUAccelTypes;
        if (!a) return false;
        for (var i = 0; i < a.length; i++) if (a[i] == t) return true;
        return false;
    }
    function benchRender(comp, label) {
        var rq = app.project.renderQueue;
        var item = rq.items.add(comp);
        var om = item.outputModule(1);
        try { om.applyTemplate("Lossless"); } catch (e) { L("applyTemplate Lossless failed: " + e); }
        om.file = new File(OUT_DIR + "/bench_" + label + ".avi");
        var t0 = new Date().getTime();
        rq.render();
        var ms = new Date().getTime() - t0;
        var frames = Math.round(comp.duration * comp.frameRate);
        L("BENCH " + label + ": " + ms + " ms for " + frames + " frames = " + (ms / frames).toFixed(1) + " ms/frame   status=" + item.status);
        try { item.remove(); } catch (e) {}
        return ms;
    }
    function setP(fx, name, v) {
        try { fx.property(name).setValue(v); } catch (e) { L("setValue " + name + " failed: " + e); }
    }

    var outFolder = new Folder(OUT_DIR);
    if (!outFolder.exists) outFolder.create();
    L("=== " + new Date().toString() + "  AE " + app.version + " (" + app.buildName + ")");

    try {
        app.newProject();
        var proj = app.project;
        var avail = app.availableGPUAccelTypes;
        L("availableGPUAccelTypes: " + (avail ? avail.join(",") : "n/a") + "  CUDA=" + GpuAccelType.CUDA + " SOFTWARE=" + GpuAccelType.SOFTWARE +
          " DIRECTX=" + (typeof GpuAccelType.DIRECTX != "undefined" ? GpuAccelType.DIRECTX : "undef") + "  initial=" + proj.gpuAccelType);

        proj.bitsPerChannel = 32;
        var comp = proj.items.addComp("FGTest", 1920, 1080, 1.0, 2.0, 24);
        var solid = comp.layers.addSolid([0.5, 0.5, 0.5], "ramp", 1920, 1080, 1.0);
        var ramp = solid.property("ADBE Effect Parade").addProperty("ADBE Ramp");
        ramp.property("ADBE Ramp-0001").setValue([0, 540]);
        ramp.property("ADBE Ramp-0002").setValue([0, 0, 0]);
        ramp.property("ADBE Ramp-0003").setValue([1920, 540]);
        ramp.property("ADBE Ramp-0004").setValue([1, 1, 1]);
        L("comp + ramp created");

        var fx = solid.property("ADBE Effect Parade").addProperty("PHUNK FastGrain");
        L("FastGrain added: name=" + fx.name + " matchName=" + fx.matchName + " numProperties=" + fx.numProperties);
        for (var i = 1; i <= fx.numProperties; i++) {
            var p = fx.property(i);
            var v = "";
            try { if (p.propertyValueType != PropertyValueType.NO_VALUE) v = " = " + p.value; } catch (e) { v = " (no value)"; }
            L("  [" + i + "] " + p.name + "  <" + p.matchName + ">" + v);
        }

        var canCUDA = hasType(GpuAccelType.CUDA);

        /* ---- CPU (software) renders ---- */
        setGPU(GpuAccelType.SOFTWARE, "SOFTWARE");
        png(comp, 0, "cpu_default_t0");
        png(comp, 5 * comp.frameDuration, "cpu_default_t5");
        setP(fx, "View", 2);
        png(comp, 0, "cpu_grainonly_t0");
        setP(fx, "View", 1);
        setP(fx, "Size", 3.0); setP(fx, "Intensity", 2.0); setP(fx, "Softness", 1.0);
        png(comp, 0, "cpu_size3_int2_soft1");
        setP(fx, "Size", 1.0); setP(fx, "Intensity", 1.0); setP(fx, "Softness", 0.5);

        /* ---- GPU renders, same frames ---- */
        if (canCUDA && setGPU(GpuAccelType.CUDA, "CUDA")) {
            png(comp, 0, "cuda_default_t0");
            png(comp, 5 * comp.frameDuration, "cuda_default_t5");
            setP(fx, "View", 2);
            png(comp, 0, "cuda_grainonly_t0");
            setP(fx, "View", 1);
            setP(fx, "Size", 3.0); setP(fx, "Intensity", 2.0); setP(fx, "Softness", 1.0);
            png(comp, 0, "cuda_size3_int2_soft1");
            setP(fx, "Size", 1.0); setP(fx, "Intensity", 1.0); setP(fx, "Softness", 0.5);
        } else {
            L("CUDA not available for GPU render test");
        }

        /* ---- benchmark: 48 frames 1080p, 16-bit ---- */
        proj.bitsPerChannel = 16;
        if (canCUDA && setGPU(GpuAccelType.CUDA, "CUDA")) benchRender(comp, "fastgrain_cuda");
        setGPU(GpuAccelType.SOFTWARE, "SOFTWARE");
        benchRender(comp, "fastgrain_cpu");

        fx.remove();
        var ag = null;
        try { ag = solid.property("ADBE Effect Parade").addProperty("VISINF Grain Implant"); } catch (e) { L("Add Grain not available: " + e); }
        if (ag) {
            L("Add Grain added: " + ag.name + " numProperties=" + ag.numProperties);
            setP(ag, "Viewing Mode", 3);
            benchRender(comp, "adobe_addgrain_cpu");
            proj.bitsPerChannel = 32;
            png(comp, 0, "adobe_addgrain_default_t0");
            ag.remove();
        }

        /* leave FastGrain applied for the user to inspect */
        proj.bitsPerChannel = 32;
        solid.property("ADBE Effect Parade").addProperty("PHUNK FastGrain");
        if (canCUDA) setGPU(GpuAccelType.CUDA, "CUDA");
        L("DONE");
    } catch (e) {
        L("ERROR: " + e.toString() + (e.line ? " (line " + e.line + ")" : ""));
    }
})();
