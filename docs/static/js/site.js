(() => {
  // The three micrograph/mask dividers share one position.
  const ranges = document.querySelectorAll(".compare-pos");
  const placeAll = (value) => ranges.forEach((range) => {
    range.value = value;
    range.closest(".compare").style.setProperty("--pos", value + "%");
  });
  ranges.forEach((range) => range.addEventListener("input", () => placeAll(range.value)));
  placeAll(50);

  // Phase-attention slides: one specimen at a time, like ZipMap's Scene State Query.
  const slides = document.querySelectorAll(".attn-slide");
  const counter = document.getElementById("attn-count");
  let current = 0;
  const turn = (step) => {
    slides[current].hidden = true;
    current = (current + step + slides.length) % slides.length;
    slides[current].hidden = false;
    counter.textContent = `${current + 1} / ${slides.length}`;
  };
  document.getElementById("attn-prev").addEventListener("click", () => turn(-1));
  document.getElementById("attn-next").addEventListener("click", () => turn(1));

  // Region-of-interest stepper. It rests on the last step and plays once when first scrolled into view.
  const roi = document.getElementById("roi");
  const layers = roi.querySelectorAll("[data-from]");
  const stepButtons = roi.querySelectorAll("[data-go]");
  const zone = roi.querySelector(".grow");
  const playButton = document.getElementById("roi-play");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FROM_CAVITY = 54.94 / 122.5;   // plastic zone starts at the cavity radius R
  const MEAN_ZONE = 121 / 122.5;       // then settles on the 94-indent mean
  let step = 7;
  let timer = null;

  const setZone = (k, animate) => {
    zone.style.transition = animate && !reduceMotion ? "" : "none";
    zone.style.setProperty("--k", k);
  };
  const show = (n) => {
    const previous = step;
    step = n;
    layers.forEach((el) => {
      const from = Number(el.dataset.from);
      const to = Number(el.dataset.to || 99);
      el.classList.toggle("on", n >= from && n <= to);
    });
    stepButtons.forEach((b) => {
      if (Number(b.dataset.go) === n) b.setAttribute("aria-current", "step");
      else b.removeAttribute("aria-current");
    });
    if (n < 5) setZone(FROM_CAVITY, false);
    else if (n === 5 && previous !== 5) {
      setZone(FROM_CAVITY, false);
      requestAnimationFrame(() => requestAnimationFrame(() => setZone(1, true)));
    } else if (n > 5) setZone(MEAN_ZONE, true);
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    playButton.textContent = step === 7 ? "Replay" : "Play";
  };
  const play = () => {
    clearInterval(timer);
    let n = 1;
    show(n);
    playButton.textContent = "Pause";
    timer = setInterval(() => {
      n += 1;
      show(n);
      if (n >= 7) stop();
    }, 2300);
  };

  stepButtons.forEach((b) => b.addEventListener("click", () => { stop(); show(Number(b.dataset.go)); stop(); }));
  playButton.addEventListener("click", () => (timer ? stop() : play()));
  show(7);
  playButton.textContent = "Replay";
  if (!reduceMotion && "IntersectionObserver" in window) {
    const watcher = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { watcher.disconnect(); play(); }
    }, { threshold: 0.6 });
    watcher.observe(roi);
  }

  const copyButton = document.getElementById("copy-bibtex");
  const bib = document.getElementById("bibtex");
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bib.textContent);
      copyButton.textContent = "Copied";
    } catch {
      const selection = getSelection();
      const span = document.createRange();
      span.selectNodeContents(bib);
      selection.removeAllRanges();
      selection.addRange(span);
      copyButton.textContent = "Selected";
    }
    setTimeout(() => { copyButton.textContent = "Copy"; }, 1800);
  });
})();
