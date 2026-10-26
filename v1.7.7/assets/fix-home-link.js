// Keep the sidebar package name inside the current docs version.
// Relative "../../" from pretty-URL pages without a trailing slash can resolve
// to the site root (/Somoclu.jl/), which some browsers still have cached as a
// 301 to an old custom domain.
document.addEventListener("DOMContentLoaded", function () {
  var a = document.querySelector(".docs-package-name a");
  if (!a) return;
  var base = typeof documenterBaseURL === "string" ? documenterBaseURL : ".";
  if (base === "" || base === ".") {
    a.setAttribute("href", "index.html");
  } else {
    a.setAttribute("href", base.replace(/\/$/, "") + "/index.html");
  }
});
