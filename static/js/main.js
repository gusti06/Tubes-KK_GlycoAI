let riskChart = null;

function setResultTheme(intensity) {
  const badge = document.getElementById('resultBadge');
  const scoreLabel = document.getElementById('riskLabel');
  const scoreValue = document.getElementById('riskScore');

  badge.classList.remove('bg-success', 'bg-warning', 'bg-danger');
  scoreLabel.classList.remove('result-low', 'result-medium', 'result-high');
  scoreValue.classList.remove('result-low', 'result-medium', 'result-high');

  if (intensity === 'high') {
    badge.classList.add('bg-danger');
    scoreLabel.classList.add('result-high');
    scoreValue.classList.add('result-high');
  } else if (intensity === 'medium') {
    badge.classList.add('bg-warning');
    scoreLabel.classList.add('result-medium');
    scoreValue.classList.add('result-medium');
  } else {
    badge.classList.add('bg-success');
    scoreLabel.classList.add('result-low');
    scoreValue.classList.add('result-low');
  }
}

function renderChart(stats) {
  const canvas = document.getElementById('riskChart');
  if (!canvas) return;

  const chartData = [stats.low || 0, stats.medium || 0, stats.high || 0];
  const chartLabels = ['Rendah', 'Sedang', 'Tinggi'];

  if (riskChart) {
    riskChart.data.datasets[0].data = chartData;
    riskChart.update();
    return;
  }

  riskChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: chartLabels,
      datasets: [{
        data: chartData,
        backgroundColor: ['#22c55e', '#f59e0b', '#fb7185'],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#edf5ff',
            font: {
              family: 'Poppins',
            },
          },
        },
      },
      cutout: '68%',
    },
  });
}

function updateStatsOnPage(stats) {
  const totalPredictions = document.getElementById('totalPredictions');
  const lowPercent = document.getElementById('lowPercent');
  const mediumPercent = document.getElementById('mediumPercent');
  const highPercent = document.getElementById('highPercent');

  if (totalPredictions) totalPredictions.textContent = stats.total ?? 0;
  if (lowPercent) lowPercent.textContent = `${stats.low_percent ?? 0}%`;
  if (mediumPercent) mediumPercent.textContent = `${stats.medium_percent ?? 0}%`;
  if (highPercent) highPercent.textContent = `${stats.high_percent ?? 0}%`;

  renderChart(stats);
}

function showLoading(show) {
  const overlay = document.getElementById('loadingOverlay');
  if (!overlay) return;
  overlay.classList.toggle('d-none', !show);
}

function setRecommendations(recommendations) {
  const list = document.getElementById('recommendationList');
  if (!list) return;
  list.innerHTML = '';
  recommendations.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    list.appendChild(li);
  });
}

function setRules(rules) {
  const explanation = document.getElementById('resultExplanation');
  if (!explanation) return;
  const topRule = rules && rules.length ? rules[0].name : '';
  if (topRule) {
    explanation.textContent = `${explanation.textContent} Aturan dominan: ${topRule}.`;
  }
}

function applyResult(result) {
  const score = Math.round(result.risk_score);
  const riskScore = document.getElementById('riskScore');
  const riskLabel = document.getElementById('riskLabel');
  const riskBar = document.getElementById('riskBar');
  const resultBadge = document.getElementById('resultBadge');
  const resultExplanation = document.getElementById('resultExplanation');

  if (riskScore) riskScore.textContent = `${score}%`;
  if (riskLabel) riskLabel.textContent = result.category;
  if (riskBar) riskBar.style.width = `${score}%`;
  if (resultBadge) resultBadge.textContent = result.category;
  if (resultExplanation) resultExplanation.textContent = result.explanation;

  setResultTheme(result.intensity);
  setRecommendations(result.recommendations || []);
  setRules(result.rule_details || []);
}

async function refreshStats() {
  const response = await fetch('/api/stats');
  const payload = await response.json();
  if (payload.success) {
    updateStatsOnPage(payload.stats);
    // Tampilkan juga statistik dari dataset mentah (dataset_overview)
    try {
      const ds = payload.dataset || {};
      const el = document.getElementById('datasetDiabetesPercent');
      if (el && typeof ds.diabetes_percent !== 'undefined') {
        el.textContent = `${ds.diabetes_percent}%`;
      }
    } catch (e) {
      // ignore
    }
  }
}

function validateForm(form) {
  let valid = true;
  const formData = new FormData(form);

  ['age', 'bmi', 'glucose', 'blood_pressure', 'insulin', 'family_history', 'activity'].forEach((field) => {
    const input = form.elements[field];
    if (!input || !input.checkValidity()) {
      valid = false;
    }
  });

  const age = Number(formData.get('age'));
  const bmi = Number(formData.get('bmi'));
  const glucose = Number(formData.get('glucose'));
  const bloodPressure = Number(formData.get('blood_pressure'));
  const insulin = Number(formData.get('insulin'));

  if (age <= 0 || bmi <= 0 || glucose < 0 || bloodPressure < 0 || insulin < 0) {
    valid = false;
  }

  form.classList.add('was-validated');
  return valid;
}

function initPredictionForm() {
  const form = document.getElementById('predictForm');
  if (!form) return;

  refreshStats();

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    if (!validateForm(form)) {
      return;
    }

    const payload = {
      age: Number(form.age.value),
      bmi: Number(form.bmi.value),
      glucose: Number(form.glucose.value),
      blood_pressure: Number(form.blood_pressure.value),
      insulin: Number(form.insulin.value),
      family_history: Number(form.family_history.value),
      activity: Number(form.activity.value),
    };

    showLoading(true);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Prediksi gagal diproses.');
      }

      applyResult(data.result);
      updateStatsOnPage(data.stats);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
      alert(error.message);
    } finally {
      showLoading(false);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initPredictionForm();
  refreshStats().catch(() => {});
});
