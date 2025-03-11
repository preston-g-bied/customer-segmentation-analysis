/**
 * Main JavaScript file for the Online Retail Analysis Dashboard
 */

// Wait for the document to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Add animation classes to elements
    addAnimations();
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Format number elements with commas, currency symbols, etc.
    formatNumbers();
});

/**
 * Add animation classes to various elements for a more engaging UI
 */
function addAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.classList.add('animate-fade-in');
        card.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Add slide-up animation to stats cards
    const statsCards = document.querySelectorAll('.stats-card');
    statsCards.forEach((card, index) => {
        card.classList.add('animate-slide-up');
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

/**
 * Format numbers in elements with data-format attributes
 */
function formatNumbers() {
    // Format currency values
    document.querySelectorAll('[data-format="currency"]').forEach(el => {
        const value = parseFloat(el.textContent);
        if (!isNaN(value)) {
            el.textContent = formatCurrency(value);
        }
    });
    
    // Format percentages
    document.querySelectorAll('[data-format="percent"]').forEach(el => {
        const value = parseFloat(el.textContent);
        if (!isNaN(value)) {
            el.textContent = formatPercent(value);
        }
    });
    
    // Format numbers with commas
    document.querySelectorAll('[data-format="number"]').forEach(el => {
        const value = parseFloat(el.textContent);
        if (!isNaN(value)) {
            el.textContent = formatNumber(value);
        }
    });
}

/**
 * Format a number as currency
 * @param {number} value - The number to format
 * @returns {string} - Formatted currency string
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-GB', {
        style: 'currency',
        currency: 'GBP',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format a number as a percentage
 * @param {number} value - The number to format
 * @returns {string} - Formatted percentage string
 */
function formatPercent(value) {
    return new Intl.NumberFormat('en-GB', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value / 100);
}

/**
 * Format a number with commas for thousands
 * @param {number} value - The number to format
 * @returns {string} - Formatted number string
 */
function formatNumber(value) {
    return new Intl.NumberFormat('en-GB').format(value);
}

/**
 * Update a Plotly chart based on user selections
 * @param {string} chartId - The ID of the chart element
 * @param {string} endpoint - The API endpoint to fetch data from
 * @param {Object} params - The parameters to send to the endpoint
 */
function updateChart(chartId, endpoint, params) {
    // Convert params object to query string
    const queryString = Object.keys(params)
        .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
        .join('&');
    
    // Fetch the data from the API
    fetch(`${endpoint}?${queryString}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update the chart
            Plotly.react(chartId, data.data, data.layout);
        })
        .catch(error => {
            console.error('Error updating chart:', error);
            // Show error message to user
            const chartElement = document.getElementById(chartId);
            if (chartElement) {
                chartElement.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i>
                        Error loading chart data. Please try again later.
                    </div>
                `;
            }
        });
}

/**
 * Toggle the visibility of a section
 * @param {string} sectionId - The ID of the section to toggle
 */
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.toggle('d-none');
        
        // Find the toggle button and update its icon
        const toggleButton = document.querySelector(`[data-toggle="${sectionId}"]`);
        if (toggleButton) {
            const icon = toggleButton.querySelector('i');
            if (icon) {
                if (section.classList.contains('d-none')) {
                    icon.classList.remove('fa-chevron-up');
                    icon.classList.add('fa-chevron-down');
                } else {
                    icon.classList.remove('fa-chevron-down');
                    icon.classList.add('fa-chevron-up');
                }
            }
        }
    }
}

/**
 * Load and display tabular data
 * @param {string} tableId - The ID of the table element
 * @param {string} endpoint - The API endpoint to fetch data from
 * @param {Array} columns - Array of column configurations
 * @param {Object} params - The parameters to send to the endpoint
 */
function loadTableData(tableId, endpoint, columns, params = {}) {
    // Convert params object to query string
    const queryString = Object.keys(params)
        .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
        .join('&');
    
    // Get the table element
    const tableElement = document.getElementById(tableId);
    if (!tableElement) {
        console.error(`Table element with ID "${tableId}" not found`);
        return;
    }
    
    // Clear existing table content
    tableElement.innerHTML = '';
    
    // Add loading indicator
    tableElement.innerHTML = `
        <tr>
            <td colspan="${columns.length}" class="text-center p-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Loading data...</p>
            </td>
        </tr>
    `;
    
    // Fetch the data from the API
    fetch(`${endpoint}?${queryString}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Clear loading indicator
            tableElement.innerHTML = '';
            
            // Check if data is empty
            if (!data || data.length === 0) {
                tableElement.innerHTML = `
                    <tr>
                        <td colspan="${columns.length}" class="text-center p-3">
                            <i class="fas fa-info-circle text-info me-2"></i>
                            No data available
                        </td>
                    </tr>
                `;
                return;
            }
            
            // Populate the table with data
            data.forEach(item => {
                const row = document.createElement('tr');
                
                columns.forEach(column => {
                    const cell = document.createElement('td');
                    
                    // Apply custom formatting if provided
                    let value = item[column.field];
                    if (column.format) {
                        switch (column.format) {
                            case 'currency':
                                value = formatCurrency(value);
                                break;
                            case 'percent':
                                value = formatPercent(value);
                                break;
                            case 'number':
                                value = formatNumber(value);
                                break;
                            case 'date':
                                value = new Date(value).toLocaleDateString();
                                break;
                            case 'custom':
                                if (column.formatter && typeof column.formatter === 'function') {
                                    value = column.formatter(value, item);
                                }
                                break;
                        }
                    }
                    
                    cell.textContent = value;
                    
                    // Apply custom classes if provided
                    if (column.cellClass) {
                        cell.className = column.cellClass;
                    }
                    
                    row.appendChild(cell);
                });
                
                tableElement.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Error loading table data:', error);
            // Show error message
            tableElement.innerHTML = `
                <tr>
                    <td colspan="${columns.length}" class="text-center p-3">
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            Error loading data. Please try again later.
                        </div>
                    </td>
                </tr>
            `;
        });
}