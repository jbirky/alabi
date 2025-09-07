// JavaScript for collapsible notebook outputs
document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers for collapsible outputs
    const collapsibleOutputs = document.querySelectorAll('.nbsphinx .nboutput-collapse');
    
    collapsibleOutputs.forEach(function(output) {
        output.addEventListener('click', function() {
            this.classList.toggle('expanded');
        });
    });
});
