//СОВЕТ: С помощью Search Everywhere вы можете найти любое действие, файл или символ в вашем проекте. Нажмите <shortcut actionId="Shift"/> <shortcut actionId="Shift"/>, введите <b>terminal</b> и нажмите <shortcut actionId="EditorEnter"/>. Затем запустите <shortcut raw="npm run dev"/> в терминале и нажмите на ссылку в его выводе, чтобы открыть приложение в браузере.
export function setupCounter(element) {
  //СОВЕТ: Попробуйте <shortcut actionId="GotoDeclaration"/> на <shortcut raw="counter"/>, чтобы увидеть его использование. Вы также можете использовать это сочетание клавиш для перехода к объявлению – попробуйте это на <shortcut raw="counter"/> в строке 13.
  let counter = 0;

  const adjustCounterValue = value => {
    if (value >= 100) return value - 100;
    if (value <= -100) return value + 100;
    return value;
  };

  const setCounter = value => {
    counter = adjustCounterValue(value);
    //СОВЕТ: WebStorm имеет множество проверок, которые помогут вам обнаружить проблемы в вашем проекте. Он также имеет быстрые исправления, которые помогут вам их решить. Нажмите <shortcut actionId="ShowIntentionActions"/> на <shortcut raw="text"/> и выберите <b>Inline variable</b>, чтобы очистить избыточный код.
    const text = `${counter}`;
    element.innerHTML = text;
  };

  document.getElementById('increaseByOne').addEventListener('click', () => setCounter(counter + 1));
  document.getElementById('decreaseByOne').addEventListener('click', () => setCounter(counter - 1));
  document.getElementById('increaseByTwo').addEventListener('click', () => setCounter(counter + 2));
  //СОВЕТ: В приложении, запущенном в браузере, вы обнаружите, что нажатие на <b>-2</b> не работает. Чтобы исправить это, перепишите его, используя код из строк 19 - 21 в качестве примеров логики.
  document.getElementById('decreaseByTwo')

  //СОВЕТ: Давайте посмотрим, как просматривать и фиксировать ваши изменения. Нажмите <shortcut actionId="GotoAction"/> и найдите <b>commit</b>. Попробуйте проверить различия для файла – дважды щелкните main.js, чтобы сделать это.
  setCounter(0);
}

//СОВЕТ: Чтобы найти текстовые строки в вашем проекте, вы можете использовать сочетание клавиш <shortcut actionId="FindInPath"/>. Нажмите его и введите <b>counter</b> – вы получите все совпадения в одном месте.
setupCounter(document.getElementById('counter-value'));

//СОВЕТ: В WebStorm есть гораздо больше возможностей, которые помогут вам быть более продуктивным. Нажмите <shortcut actionId="Shift"/> <shortcut actionId="Shift"/> и найдите <b>Learn WebStorm</b>, чтобы открыть наш учебный центр с дополнительными возможностями для вас.
