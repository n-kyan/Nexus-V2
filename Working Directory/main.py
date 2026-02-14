import flet as ft
from mlx_lm import load, stream_generate
import gc, asyncio

# Global State - Kept outside main to persist across sessions if needed
state = {
    "model": None, 
    "tokenizer": None, 
    "messages": [], # The active context
    "summary": ""   # The "compressed" long-term memory
}

MODELS = {
    "Daily Driver (Llama 3.2 3B)": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "Coding Expert (Qwen 2.5 7B)": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "Vision/Omni (Qwen 3 VL)": "mlx-community/Qwen3-VL-8B-Instruct-4bit"
}

async def main(page: ft.Page):
    page.title = "Nexus M4 Engine"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20

    chat_view = ft.Column(expand=True, scroll=ft.ScrollMode.ALWAYS, spacing=10)
    user_input = ft.TextField(hint_text="Ask Nexus...", expand=True, shift_enter=True)
    loader = ft.ProgressBar(visible=False, color=ft.Colors.BLUE_ACCENT)
    status_text = ft.Text("Ready", size=12, color="grey")

    async def load_model_task(model_path):
        state["model"], state["tokenizer"] = None, None
        gc.collect() 
        loader.visible = True
        status_text.value = f"Loading {model_path}..."
        page.update()
        
        state["model"], state["tokenizer"] = load(model_path)
        loader.visible = False
        status_text.value = f"Active: {model_path}"
        page.update()

    async def clear_and_compress(e):
        if not state["messages"]: return
        status_text.value = "Compressing context..."
        page.update()

        # 1. Ask model to summarize before wiping
        summary_prompt = "Summarize our entire conversation so far into one dense paragraph. Focus on the goals and current progress."
        temp_messages = state["messages"] + [{"role": "user", "content": summary_prompt}]
        prompt = state["tokenizer"].apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
        
        # We generate the summary quickly (no streaming needed for the background task)
        summary_text = ""
        for response in stream_generate(state["model"], state["tokenizer"], prompt):
            summary_text += response.text

        # 2. Wipe history but keep the summary
        state["summary"] = summary_text
        state["messages"] = [{"role": "system", "content": f"Previous conversation summary: {summary_text}"}]
        
        chat_view.controls.clear()
        chat_view.controls.append(ft.Text(f"Context Compressed: {summary_text[:100]}...", italic=True, color="grey"))
        status_text.value = "Memory Optimized."
        page.update()

    async def start_generation(e):
        if not user_input.value or state["model"] is None: return
        
        user_text = user_input.value
        user_input.value = ""
        state["messages"].append({"role": "user", "content": user_text})
        
        # UI Bubbles
        chat_view.controls.append(ft.Container(content=ft.Text(user_text), alignment=ft.Alignment.CENTER_RIGHT, bgcolor=ft.Colors.BLUE_900, padding=10, border_radius=10))
        ai_response_text = ft.Text("", color=ft.Colors.GREEN_300)
        chat_view.controls.append(ft.Container(content=ai_response_text, alignment=ft.Alignment.CENTER_LEFT, bgcolor=ft.Colors.GREY_900, padding=10, border_radius=10))
        page.update()

        prompt = state["tokenizer"].apply_chat_template(state["messages"], tokenize=False, add_generation_prompt=True)

        full_response = ""
        for response in stream_generate(state["model"], state["tokenizer"], prompt):
            full_response += response.text
            ai_response_text.value = full_response
            page.update()
            await asyncio.sleep(0)

        state["messages"].append({"role": "assistant", "content": full_response})

    page.add(
        ft.Row([
            ft.Text("Nexus AI", size=24, weight="bold"),
            ft.Dropdown(label="Engine", options=[ft.DropdownOption(k) for k in MODELS.keys()], on_select=lambda e: page.run_task(load_model_task, MODELS[e.data]), width=250),
            ft.ElevatedButton("Compress Chat", icon=ft.Icons.REPLY_ALL, on_click=lambda e: page.run_task(clear_and_compress))
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        loader, status_text, ft.Divider(), chat_view,
        ft.Row([user_input, ft.IconButton(icon=ft.Icons.PLAY_ARROW_ROUNDED, on_click=lambda e: page.run_task(start_generation, e))])
    )

if __name__ == "__main__":
    ft.run(main)