const ctx = document.getElementById('myChart').getContext('2d');

fetch('/api/schemas-data?scheme=your_scheme_here')
    .then(response => response.json())
    .then(data => {
        const labels = data.map(item => item.label);
        const values = data.map(item => item.value);

        const myChart = new Chart(ctx, {
            type: 'bar', // Change to 'line', 'pie', etc. as needed
            data: {
                labels: labels,
                datasets: [{
                    label: 'Dataset Label',
                    data: values,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    })
    .catch(error => console.error('Error fetching data:', error));