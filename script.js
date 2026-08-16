const header = document.querySelector("[data-header]");
const navToggle = document.querySelector(".nav-toggle");
const nav = document.querySelector(".site-nav");

const updateHeader = () => {
  header?.classList.toggle("scrolled", window.scrollY > 24);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle?.addEventListener("click", () => {
  const isOpen = nav?.classList.toggle("open") ?? false;
  navToggle.setAttribute("aria-expanded", String(isOpen));
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

const revealElements = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8%", threshold: 0.08 },
  );
  revealElements.forEach((element) => observer.observe(element));
} else {
  revealElements.forEach((element) => element.classList.add("visible"));
}

const copyButton = document.querySelector("[data-copy-citation]");
copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex")?.textContent?.trim();
  if (!citation) return;

  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = "Copied";
  } catch {
    copyButton.textContent = "Select and copy";
  }

  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1800);
});
