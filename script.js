const form = document.getElementById("todo-form");
const input = document.getElementById("todo-input");
const list = document.getElementById("todo-list");
const taskCount = document.getElementById("task-count");
const clearCompletedBtn = document.getElementById("clear-completed");

let todos = JSON.parse(localStorage.getItem("todos") || "[]");

function saveTodos() {
  localStorage.setItem("todos", JSON.stringify(todos));
}

function updateCount() {
  const remaining = todos.filter((t) => !t.done).length;
  taskCount.textContent = `${remaining} task${remaining === 1 ? "" : "s"} left`;
}

function render() {
  list.innerHTML = "";
  todos.forEach((todo, index) => {
    const li = document.createElement("li");
    li.className = "todo-item" + (todo.done ? " done" : "");

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = todo.done;
    checkbox.addEventListener("change", () => toggleTodo(index));

    const span = document.createElement("span");
    span.textContent = todo.text; // textContent prevents HTML injection

    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "×";
    deleteBtn.className = "delete-btn";
    deleteBtn.setAttribute("aria-label", "Delete task");
    deleteBtn.addEventListener("click", () => deleteTodo(index));

    li.append(checkbox, span, deleteBtn);
    list.appendChild(li);
  });
  updateCount();
}

function addTodo(text) {
  todos.push({ text, done: false });
  saveTodos();
  render();
}

function toggleTodo(index) {
  todos[index].done = !todos[index].done;
  saveTodos();
  render();
}

function deleteTodo(index) {
  todos.splice(index, 1);
  saveTodos();
  render();
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addTodo(text);
  input.value = "";
});

clearCompletedBtn.addEventListener("click", () => {
  todos = todos.filter((t) => !t.done);
  saveTodos();
  render();
});

render();