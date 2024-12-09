function performSearch() {
    const resultsDiv = document.getElementById("results");

    // Get form data
    const formData = new FormData();
    const imageFile = document.getElementById("image-query").files[0];
    const textQuery = document.getElementById("text-query").value;
    const queryWeight = document.getElementById("query-weight").value;
    const queryType = document.getElementById("query-type").value;
    const usePCA = document.getElementById("use-pca").checked; // Get PCA toggle state
    const pcaK = parseInt(document.getElementById("pca-k").value, 10) || 10;

    if (imageFile) formData.append("image-query", imageFile);
    formData.append("text-query", textQuery);
    formData.append("query-weight", queryWeight);
    formData.append("query-type", queryType);
    formData.append("use-pca", usePCA);
    if (usePCA) formData.append("pca-k", pcaK); // Add k only if PCA is enabled

    // Clear existing results and show loading
    resultsDiv.innerHTML = "<p>Searching... (This is where the results will appear)</p>";

    // Send request to Flask backend
    fetch("/search", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((results) => {
            if (results.error) {
                resultsDiv.innerHTML = `<p style="color: red;">${results.error}</p>`;
            } else {
                // Clear and display results
                resultsDiv.innerHTML = `<h3>Search Results:</h3>`;
                results.forEach((result) => {
                    resultsDiv.innerHTML += `
                        <div>
                            <img src="${result.image}" alt="Result Image">
                            <p class="similarity">Similarity: ${result.similarity.toFixed(2)}</p>
                        </div>
                    `;
                });
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            resultsDiv.innerHTML = `<p>Error occurred while searching.</p>`;
        });
    
}
