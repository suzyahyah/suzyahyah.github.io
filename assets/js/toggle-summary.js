const toggleButton = document.getElementById("toggleGptSummary");
const toggledElements = document.querySelectorAll(".gptSummary");

// Initial state: all elements are hidden
let allHidden = true;

toggleButton.addEventListener("click", function() {
    // Toggle the visibility of all elements with the class "toggledElement"
    for (const element of toggledElements) {
        if (allHidden) {
            element.style.display = "block";
        } else {
            element.style.display = "none";
        }
    }

    // Toggle the state for the next click
    allHidden = !allHidden;
});

