/*
 * motion_bridge.jsx — After Effects ↔ motion CLI proof-of-concept.
 *
 * Runs `motion merge` on two stills, imports the resulting transition video, and
 * reads the JSON manifest so AE knows the object list and per-segment frame ranges.
 *
 * Setup:
 *   - Edit CONFIG below (paths to your two images + the `motion` binary).
 *   - AE ▸ Preferences ▸ Scripting & Expressions ▸ "Allow Scripts to Write Files
 *     and Access Network".
 *   - File ▸ Scripts ▸ Run Script File… ▸ pick this file.
 *
 * This is intentionally small: the real contract is the manifest, not this script.
 */

(function () {
    var CONFIG = {
        MOTION_BIN: $.getenv("MOTION_BIN") || "motion", // or absolute path to .venv/bin/motion
        imageA: "~/coding/motion/outputs/smoke/a.png",
        imageB: "~/coding/motion/outputs/smoke/b.png",
        outVideo: "~/coding/motion/outputs/ae_merge.mp4",
        promptA: "",
        promptB: "",
        frames: 48,
        fps: 30
    };

    function sh(cmd) {
        // Wrap in /bin/sh so PATH/venv resolution and quoting behave like a terminal.
        return system.callSystem('/bin/sh -c "' + cmd.replace(/"/g, '\\"') + '"');
    }

    function readFile(path) {
        var f = new File(path);
        if (!f.exists) return null;
        f.open("r");
        var txt = f.read();
        f.close();
        return txt;
    }

    function buildCommand(c) {
        var parts = [
            c.MOTION_BIN, "merge",
            "'" + c.imageA + "'", "'" + c.imageB + "'",
            "-o", "'" + c.outVideo + "'",
            "--frames", c.frames,
            "--fps", c.fps,
            "--write-frames"
        ];
        if (c.promptA) parts.push("--prompt-a", "'" + c.promptA + "'");
        if (c.promptB) parts.push("--prompt-b", "'" + c.promptB + "'");
        return parts.join(" ");
    }

    app.beginUndoGroup("motion merge");
    try {
        var cmd = buildCommand(CONFIG);
        $.writeln("running: " + cmd);
        var out = sh(cmd);
        $.writeln(out);

        // Manifest lives next to the video: out.mp4 -> out.json
        var manifestPath = CONFIG.outVideo.replace(/\.[^.]+$/, ".json");
        var manifestTxt = readFile(manifestPath);
        var manifest = manifestTxt ? JSON.parse(manifestTxt) : null;

        // Import the rendered transition.
        var io = new ImportOptions(new File(CONFIG.outVideo));
        var footage = app.project.importFile(io);

        // Drop it into a comp and, if we have a manifest, name/annotate from it.
        var fps = manifest ? manifest.fps : CONFIG.fps;
        var comp = app.project.items.addComp(
            "motion_merge", footage.width, footage.height, 1.0,
            Math.max(footage.duration, 1), fps
        );
        var layer = comp.layers.add(footage);

        if (manifest) {
            var names = [];
            for (var i = 0; i < manifest.objects.length; i++) {
                names.push(manifest.objects[i].prompt || ("obj" + i));
            }
            layer.name = "morph: " + names.join(" → ");
            $.writeln("segments: " + manifest.segments.length +
                      "  total_frames: " + manifest.total_frames);
        }
        comp.openInViewer();
    } catch (e) {
        alert("motion bridge error:\n" + e.toString());
    } finally {
        app.endUndoGroup();
    }
})();
