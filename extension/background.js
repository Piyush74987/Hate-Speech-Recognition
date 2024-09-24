chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed.");
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    console.log("Page loaded:", tab.url);
    console.log("Tab ID:", tabId);
    if (tab.url === "chrome://newtab/") return;
    performFetchHateSpeech(tab.url, tab.title);
    performFetchContext(tab.url);
  }
});

/**
 * @param {string} url
 * @param {string} pageTitle
 * @returns void
 */
function performFetchHateSpeech(url, pageTitle) {
  fetch(`http://localhost:8000/hate_speech?url=${url}`, {
    method: "GET",
  })
    .then((res) => {
      return res.json();
    })
    .then((data) => {
      if (data.hate_speech) {
        const pageMessage = "Hate Speech Detected on: " + pageTitle;
        showNotification("Hate Speech Detected", pageMessage);
      }
    })
    .catch((err) => console.log(err));
}

/**
 * @param {string} url
 * @param {string} pageTitle
 * @param {string} pageMessage
 * @returns void
 */
function performFetchContext(url, pageTitle) {
  fetch(`http://localhost:8000/context_detection?url=${url}`, {
    method: "GET",
  })
    .then((res) => {
      return res.json();
    })
    .then((data) => {
      console.log(data);
      if (data.context) {
        const pageMessage = "Page contains context: " + data.context;
        showNotification("Context Detected on:" + pageTitle, pageMessage);
      }
    })
    .catch((err) => console.log(err));
}

/**
 * @param {string} pageTitle
 * @param {string} pageMessage
 * @returns void
 */
// showNotification displays a notifcation for the user
function showNotification(pageTitle, pageMessage) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "./public/android-chrome-192x192.png",
    title: pageTitle,
    message: pageMessage,
    priority: 2,
  });
}
