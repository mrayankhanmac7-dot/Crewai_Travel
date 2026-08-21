import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(page_title="AI Travel Crew", page_icon="✈️", layout="wide")

# ---- the key comes from Streamlit secrets, never from the code ----
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("No API key found. Add OPENAI_API_KEY in Settings -> Secrets.")
    st.stop()

llm = LLM(model="openai/gpt-4o-mini", temperature=0.4)

MAX_RUNS = 5

STYLES = {
    "Balanced":   "You suggest a good mix of famous sights and local experiences.",
    "Foodie":     "You are obsessed with food. Every day must revolve around eating.",
    "Adventure":  "You love the outdoors. You favour hiking, water and adrenaline.",
    "Budget":     "You care strongly about keeping costs down. You find free and cheap options.",
    "Tech & Science": "You love science museums, technology and anything futuristic.",
}


def build_crew(destination, days, style):
    researcher = Agent(
        role="Teen Travel Researcher",
        goal="Find fun and age-appropriate travel activities for teenagers",
        backstory=(
            "You are curious, practical, and great at discovering activities "
            "that teenagers would genuinely enjoy. " + STYLES[style]
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )

    planner = Agent(
        role="Teen Trip Planner",
        goal="Turn travel ideas into a fun, realistic and organised itinerary",
        backstory=(
            "You are an organised trip planner who understands that teenagers "
            "want variety, fun, reasonable pacing, and time to relax. You "
            "never cram too much into one day."
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )

    research_task = Task(
        description=(
            f"Recommend 5 exciting, age-appropriate activities for teenagers "
            f"visiting {destination}. For each: a short name, what they would "
            f"do, and why it is fun."
        ),
        expected_output="A numbered list of exactly 5 activities with short explanations.",
        agent=researcher,
    )

    planning_task = Task(
        description=(
            f"Using the research, build a {days}-day itinerary for teenagers "
            f"visiting {destination}. Each day needs Morning, Afternoon and "
            f"Evening. Keep it realistic - travel time is real."
        ),
        expected_output=(
            f"A clear {days}-day itinerary, each day split into Morning, "
            "Afternoon and Evening."
        ),
        agent=planner,
        context=[research_task],
    )

    return researcher, planner, research_task, planning_task


def run_one(agent, task):
    return str(Crew(agents=[agent], tasks=[task],
                    process=Process.sequential, verbose=False).kickoff())


for k, v in [("runs", 0), ("trip", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---- sidebar ----
with st.sidebar:
    st.markdown("### 👥 Your crew")
    st.markdown("**🔎 Researcher** — finds the activities\n\n"
                "**🗺️ Planner** — builds the itinerary")
    st.divider()
    days = st.slider("📅 How many days?", 1, 5, 3)
    style = st.selectbox("🎒 Travel style", list(STYLES))
    st.caption(STYLES[style])
    st.divider()
    st.markdown(f"**Runs left:** {MAX_RUNS - st.session_state.runs}")
    if st.session_state.trip and st.button("🔄 Start over", use_container_width=True):
        st.session_state.trip = None
        st.rerun()

# ---- main ----
st.title("✈️ AI Travel Crew")
st.caption("Two AI agents plan a trip for teenagers, anywhere in the world.")

destination = st.text_input(
    "Where should the crew plan a trip?",
    placeholder="e.g. Japan, Turkey, Morocco, Scotland",
    max_chars=60,
)

if st.button("🚀 Plan my trip", type="primary", use_container_width=True):
    if st.session_state.runs >= MAX_RUNS:
        st.error("You have used all your runs. Refresh the page to reset.")
    elif len(destination.strip()) < 2:
        st.warning("Type a destination first.")
    else:
        d = destination.strip()
        try:
            researcher, planner, r_task, p_task = build_crew(d, days, style)

            with st.status("🔎 Researcher is finding activities...") as s:
                research = run_one(researcher, r_task)
                s.update(label="🔎 Researcher found 5 activities", state="complete")

            # hand the research to the planner by hand
            p_task.description += f"\n\nTHE RESEARCH:\n{research}"

            with st.status("🗺️ Planner is building the itinerary...") as s:
                itinerary = run_one(planner, p_task)
                s.update(label="🗺️ Itinerary is ready", state="complete")

            st.session_state.runs += 1
            st.session_state.trip = {
                "destination": d, "days": days, "style": style,
                "research": research, "itinerary": itinerary,
            }
            st.rerun()

        except Exception as e:
            st.error("Something went wrong.")
            st.caption(f"{type(e).__name__}: {e}")

t = st.session_state.trip
if t:
    st.divider()
    st.subheader(f"🌍 {t['destination']} · {t['days']} days · {t['style']}")

    tab1, tab2 = st.tabs(["🗺️ Your itinerary", "🔎 The research"])
    with tab1:
        st.markdown(t["itinerary"])
    with tab2:
        st.markdown(t["research"])

    st.download_button(
        "⬇️ Download the trip plan",
        data=(f"TRIP TO {t['destination'].upper()} ({t['days']} days, {t['style']})\n\n"
              f"=== ITINERARY ===\n{t['itinerary']}\n\n"
              f"=== RESEARCH ===\n{t['research']}"),
        file_name=f"trip_{t['destination'].replace(' ', '_')}.txt",
        mime="text/plain",
    )
