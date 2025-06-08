// Override the webdriver property
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
}); 