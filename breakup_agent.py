import streamlit as st
import logging
import tempfile
import os
import io
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass

from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage
from agno.tools.duckduckgo import DuckDuckGoTools
from docx import Document
from pypdf import PdfReader


@dataclass
class AgentConfig:
    """Configuration for each recovery agent"""
    name: str
    emoji: str
    role: str
    description: str
    instructions: List[str]
    use_tools: bool = False


class RecoverySquadApp:
    """Main application class for the Breakup Recovery Squad"""
    
    # Constants
    SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
    SUPPORTED_TEXT_TYPES = {".txt", ".docx", ".pdf"}
    MAX_TEXT_LENGTH = 4000
    MAX_PDF_PAGES = 5
    RATE_LIMIT_DELAY = 1.5
    
    # Agent configurations with human-like instructions
    AGENT_CONFIGS = {
        "Maya": AgentConfig(
            name="Maya - Your Empathetic Friend",
            emoji="💙",
            role="The Heart Healer",
            description="Emotional support & empathy",
            instructions=[
                "You are Maya, a warm and caring friend who happens to be a relationship counselor. You talk like you're having coffee with a close friend - natural, supportive, and real.",
                "",
                "YOUR PERSONALITY:",
                "- Speak like a caring friend, not a clinical therapist",
                "- Use 'I hear you', 'That sounds really tough', 'I get it' instead of formal language",
                "- Share relatable insights like 'I've seen this before' or 'Many people feel this way'",
                "- Be warm but honest - don't sugarcoat everything",
                "- Keep responses conversational and under 150 words",
                "",
                "HOW TO RESPOND:",
                "- First, acknowledge their pain with genuine empathy",
                "- Then offer ONE practical insight or gentle perspective",
                "- End with encouragement or a small actionable step",
                "- Ask ONE follow-up question if needed",
                "",
                "AVOID:",
                "- Long lists or bullet points",
                "- Clinical jargon or overly formal language", 
                "- Multiple suggestions at once",
                "- Saying 'I understand' without showing you actually do",
                "",
                "Remember: You're their friend who cares, not a textbook. Be human, be real, be brief."
            ]
        ),
        "Alex": AgentConfig(
            name="Alex - The Closure Guide",
            emoji="✨",
            role="The Peace Maker",
            description="Closure & letting go",
            instructions=[
                "You are Alex, someone who's been through heartbreak and knows how to help people find peace. You speak from experience, not theory.",
                "",
                "YOUR APPROACH:",
                "- Talk like someone who's been there and understands the pain",
                "- Focus on ONE specific way to find closure, not a whole list",
                "- Be gentle but direct about what needs to be done",
                "- Keep it under 100 words - closure advice should be clear and simple",
                "",
                "WHAT TO OFFER:",
                "- Either help write ONE unsent message OR suggest ONE ritual",
                "- Give them permission to feel their emotions",
                "- Remind them that closure comes from within, not from the other person",
                "",
                "SPEAK LIKE:",
                "'The hardest part about letting go is...' or 'What helped me was...' or 'Here's what I wish someone had told me...'",
                "",
                "Keep it real, keep it short, keep it healing."
            ]
        ),
        "Jordan": AgentConfig(
            name="Jordan - Your Recovery Coach",
            emoji="🚀",
            role="The Momentum Builder",
            description="Recovery plans & motivation",
            instructions=[
                "You are Jordan, an upbeat friend who's great at getting people back on track after tough times. You're practical and motivating.",
                "",
                "YOUR STYLE:",
                "- Sound like an encouraging friend, not a life coach robot",
                "- Give them 2-3 specific things to do TODAY, not a whole week plan",
                "- Focus on small wins that build momentum",
                "- Keep it under 120 words",
                "",
                "WHAT TO SUGGEST:",
                "- Pick ONE self-care activity for today",
                "- ONE way to avoid social media stalking",
                "- ONE small thing to feel accomplished",
                "",
                "TONE:",
                "'Alright, let's get you moving forward' or 'Here's what's gonna help right now' or 'Trust me on this one'",
                "",
                "Make it feel achievable, not overwhelming. You're their cheerleader, not their drill sergeant."
            ]
        ),
        "Sam": AgentConfig(
            name="Sam - The Truth Teller",
            emoji="💪",
            role="The Reality Check Friend",
            description="Honest truth & reality checks",
            instructions=[
                "You are Sam, the friend who loves them enough to tell the truth. You're direct but caring, like a older sibling who's seen it all.",
                "",
                "YOUR APPROACH:",
                "- Be honest without being cruel",
                "- Point out what they might not want to hear, but need to",
                "- Keep it under 100 words - truth hits harder when it's concise",
                "- End with hope or a reason why this is actually good for them",
                "",
                "HOW TO DELIVER TRUTH:",
                "- Start with 'Look, I care about you, so I'm gonna be straight with you...'",
                "- Point out ONE key thing they're not seeing clearly",
                "- Explain why moving on is the right choice",
                "- Remind them of their worth",
                "",
                "You're tough love, not mean. You're the friend who helps them see reality so they can heal."
            ],
            use_tools=True
        )
    }

    def __init__(self):
        """Initialize the application"""
        self._setup_logging()
        self._setup_page_config()
        self._initialize_session_state()

    def _setup_logging(self):
        """Configure logging for error handling"""
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        if "error_logs" not in st.session_state:
            st.session_state.error_logs = []
        
        # Add custom handler for Streamlit
        if not any(isinstance(h, self.StreamlitErrorHandler) for h in self.logger.handlers):
            handler = self.StreamlitErrorHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

    class StreamlitErrorHandler(logging.Handler):
        """Custom logging handler for Streamlit error display"""
        def emit(self, record: logging.LogRecord) -> None:
            try:
                if record.levelno >= logging.ERROR:
                    st.session_state.error_logs.append(self.format(record))
            except Exception:
                pass

    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="💔 Breakup Recovery Squad",
            page_icon="💔",
            layout="wide"
        )

    def _initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "chat_messages": [],
            "agents_initialized": False,
            "api_key_input": "",
            "selected_agent": "All Squad Members",
            "search_results": [],
            "show_agent_info": False,
            "previous_selected_agent": None
        }
        
        # Initialize agents
        for agent_key in ["therapist_agent", "closure_agent", "routine_planner_agent", "brutal_honesty_agent"]:
            defaults[agent_key] = None
            
        # Set defaults if not already set
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def handle_file_uploads(self, files) -> Tuple[List, str]:
        """Process uploaded files and return images and text content"""
        image_files = []
        texts = []

        if not files:
            return image_files, ""

        for file in files:
            ext = Path(file.name).suffix.lower()
            
            try:
                if ext in self.SUPPORTED_IMAGE_TYPES:
                    image_files.append(file)
                    st.image(file, caption=file.name, use_container_width=True)
                    
                elif ext == ".txt":
                    text = file.getvalue().decode("utf-8", errors="ignore")
                    texts.append(f"{file.name}:\n{text}")
                    with st.expander(f"Preview (txt): {file.name}"):
                        st.text(text[:2000])
                        
                elif ext == ".docx":
                    doc = Document(io.BytesIO(file.getvalue()))
                    text = "\n".join(p.text for p in doc.paragraphs)
                    texts.append(f"{file.name}:\n{text}")
                    with st.expander(f"Preview (docx): {file.name}"):
                        st.text(text[:2000])
                        
                elif ext == ".pdf":
                    reader = PdfReader(io.BytesIO(file.getvalue()))
                    pages_text = []
                    for page in reader.pages[:self.MAX_PDF_PAGES]:
                        pages_text.append(page.extract_text() or "")
                    text = "\n".join(pages_text)
                    texts.append(f"{file.name}:\n{text}")
                    st.caption(f"PDF uploaded: {file.name}")
                    
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    
            except Exception as e:
                self.logger.error(f"Error handling file {file.name}: {e}")

        text_blob = "\n\n".join(texts)
        if len(text_blob) > self.MAX_TEXT_LENGTH:
            text_blob = text_blob[:self.MAX_TEXT_LENGTH]
            
        return image_files, text_blob

    def process_images(self, files) -> List[AgnoImage]:
        """Convert uploaded image files to AgnoImage objects"""
        processed = []
        
        for file in files or []:
            try:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"temp_{file.name}")
                
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                    
                processed.append(AgnoImage(filepath=Path(temp_path)))
                
            except Exception as e:
                self.logger.error(f"Error processing image {getattr(file, 'name', 'unknown')}: {e}")
                continue
                
        return processed

    def initialize_agents(self, api_key: str) -> Tuple[Agent, ...]:
        """Initialize all recovery squad agents"""
        try:
            model = Gemini(id="gemini-2.0-flash-exp", api_key=api_key)
            agents = []
            
            for config in self.AGENT_CONFIGS.values():
                tools = [DuckDuckGoTools()] if config.use_tools else None
                
                agent = Agent(
                    model=model,
                    name=config.name,
                    instructions=config.instructions,
                    tools=tools,
                    markdown=True
                )
                agents.append(agent)
                
            return tuple(agents)
            
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            return tuple([None] * len(self.AGENT_CONFIGS))

    def get_agent_response(self, selected_agent: str, message: str, images: List = None) -> Dict[str, str]:
        """Get responses from selected agent(s) with human-like context"""
        agent_map = {
            "Maya": st.session_state.therapist_agent,
            "Alex": st.session_state.closure_agent,
            "Jordan": st.session_state.routine_planner_agent,
            "Sam": st.session_state.brutal_honesty_agent
        }
        
        responses = {}
        
        if selected_agent == "All Squad Members":
            # Group conversation - each agent responds naturally
            for i, (agent_name, agent) in enumerate(agent_map.items()):
                if agent:
                    try:
                        if i > 0:  # Rate limiting
                            time.sleep(self.RATE_LIMIT_DELAY)
                        
                        # More natural group conversation prompt
                        contextual_message = (
                            f"You're in a group chat helping someone through a breakup. "
                            f"Respond naturally as {agent_name} would - keep it brief and authentic.\n\n{message}"
                        )
                        
                        response = agent.run(message=contextual_message, images=images or [])
                        responses[agent_name] = response.content
                        
                    except Exception as e:
                        responses[agent_name] = self._handle_agent_error(agent_name, e)
        else:
            # Individual conversation - more personal and detailed
            if selected_agent in agent_map and agent_map[selected_agent]:
                try:
                    contextual_message = (
                        f"You're having a private conversation with someone going through a breakup. "
                        f"Respond as their close friend {selected_agent} would - be personal and supportive.\n\n{message}"
                    )
                    
                    response = agent_map[selected_agent].run(message=contextual_message, images=images or [])
                    responses[selected_agent] = response.content
                    
                except Exception as e:
                    responses[selected_agent] = self._handle_agent_error(selected_agent, e)
                    
        return responses

    def _handle_agent_error(self, agent_name: str, error: Exception) -> str:
        """Handle agent response errors gracefully"""
        if "429" in str(error) or "Too Many Requests" in str(error):
            return f"I'm getting a lot of requests right now. Try selecting me individually for a more personal chat! 💫"
        else:
            self.logger.error(f"Error getting response from {agent_name}: {str(error)}")
            return "I'm having trouble responding right now. Please try again in a moment."

    def search_recovery_resources(self, query: str) -> List[Dict[str, str]]:
        """Search for recovery resources using DuckDuckGo"""
        try:
            # Use Sam's agent since it has DuckDuckGo tools
            if st.session_state.brutal_honesty_agent:
                search_prompt = f"""
                Search for helpful resources, articles, or information about: {query}
                
                Focus on:
                - Reputable mental health resources
                - Relationship recovery advice
                - Scientific articles about breakup recovery
                - Self-help strategies
                
                Provide a brief summary of the most helpful results.
                """
                
                response = st.session_state.brutal_honesty_agent.run(message=search_prompt)
                
                # Parse the response to extract useful information
                return [{
                    "title": "Recovery Resources Search Results",
                    "content": response.content,
                    "source": "DuckDuckGo Search via Recovery Squad"
                }]
            else:
                return [{"title": "Search Unavailable", "content": "Please initialize agents first.", "source": "System"}]
                
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return [{"title": "Search Error", "content": f"Unable to search at this time: {str(e)}", "source": "System"}]

    def render_sidebar(self):
        """Render the sidebar with API configuration and agent selection"""
        with st.sidebar:
            st.header("🔑 API Configuration")
            
            # API Key Input
            api_key = st.text_input(
                "Enter your GEMINI_API_KEY",
                value=st.session_state.api_key_input,
                type="password",
                help="Get your API key from Google AI Studio",
                key="api_key_widget"
            )
            
            # Handle API key changes
            if api_key != st.session_state.api_key_input:
                st.session_state.api_key_input = api_key
                st.session_state.agents_initialized = False
            
            # API Key Status
            if api_key:
                st.success("API key loaded successfully ✅")
                self._initialize_agents_if_needed(api_key)
            else:
                st.warning("Please provide the API key to proceed")
                self._show_api_key_instructions()
            
            # Agent Selection Section
            if st.session_state.agents_initialized:
                self._render_agent_selection()
                self._render_search_section()
                
                if st.button("🆕 Start New Chat", type="secondary"):
                    self._start_new_chat()

    def _initialize_agents_if_needed(self, api_key: str):
        """Initialize agents if not already done"""
        if not st.session_state.agents_initialized:
            with st.spinner("Initializing your recovery squad..."):
                agents = self.initialize_agents(api_key)
                if all(agents):
                    (st.session_state.therapist_agent, 
                     st.session_state.closure_agent, 
                     st.session_state.routine_planner_agent, 
                     st.session_state.brutal_honesty_agent) = agents
                    st.session_state.agents_initialized = True
                    st.success("Recovery squad ready! 💪")

    def _show_api_key_instructions(self):
        """Show instructions for getting API key"""
        st.markdown("""
        To get your API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enable the Generative Language API in your [Google Cloud Console](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com)
        """)

    def _render_agent_selection(self):
        """Render agent selection interface"""
        st.divider()
        
        # Header with info button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("💬 Chat Settings")
        with col2:
            if st.button("ℹ️", help="Know Your Agents", key="agent_info_btn"):
                st.session_state.show_agent_info = not st.session_state.get("show_agent_info", False)
        
        # Agent information popup
        if st.session_state.get("show_agent_info", False):
            self._show_agent_info()
        
        # Agent selection dropdown
        agent_options = ["All Squad Members"] + list(self.AGENT_CONFIGS.keys())
        
        selected_agent = st.selectbox(
            "Choose who to chat with:",
            options=agent_options,
            index=agent_options.index(st.session_state.selected_agent) if st.session_state.selected_agent in agent_options else 0,
            help="Select an agent for personalized guidance or choose 'All Squad Members' for group support."
        )
        
        st.session_state.selected_agent = selected_agent
        
        # Show agent description
        self._show_agent_description(selected_agent)

    def _show_agent_info(self):
        """Show detailed agent information"""
        with st.expander("🌟 Meet Your Recovery Squad", expanded=True):
            for agent_name, config in self.AGENT_CONFIGS.items():
                st.markdown(f"""
                **{config.emoji} {config.name}**  
                *{config.role}*  
                {config.description}
                """)
            
            st.markdown("""
            **💝 All Squad Members**  
            Get comprehensive support from everyone! Each brings their unique perspective for well-rounded guidance.
            """)

    def _show_agent_description(self, selected_agent: str):
        """Show description for selected agent"""
        if selected_agent == "All Squad Members":
            st.caption("💝 Get support from everyone")
        elif selected_agent in self.AGENT_CONFIGS:
            config = self.AGENT_CONFIGS[selected_agent]
            st.caption(f"{config.emoji} {config.description}")

    def _start_new_chat(self):
        """Start a new chat session"""
        st.session_state.chat_messages = []
        st.session_state.search_results = []
        st.success("✨ New chat started! Your recovery squad is ready to help.")
        st.rerun()

    def _render_search_section(self):
        """Render the search functionality section"""
        st.divider()
        st.header("🔍 Search Resources")
        
        search_query = st.text_input(
            "Search for recovery resources:",
            placeholder="e.g., 'dealing with anxiety after breakup', 'self-care tips', 'closure techniques'"
        )
        
        if st.button("🔍 Search Recovery Resources", type="secondary"):
            if search_query:
                with st.spinner("Searching for helpful resources..."):
                    results = self.search_recovery_resources(search_query)
                    st.session_state.search_results = results
                    
                    if results:
                        st.success(f"Found resources for: '{search_query}'")
                        for result in results:
                            with st.expander(f"📚 {result['title']}"):
                                st.markdown(result['content'])
                                st.caption(f"Source: {result['source']}")
            else:
                st.warning("Please enter a search query")

    def render_main_content(self):
        """Render the main application content"""
        st.title("💔 Breakup Recovery Squad")
        st.markdown("""
        ### Your friends who get it are here to help
        Tell us what happened, share your chat screenshots - we'll help you through this tough time with real talk and genuine care.
        """)
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("What's going on?")
            user_input = st.text_area(
                "Tell us your story - we're here to listen",
                height=150,
                placeholder="What happened? How are you feeling? Share as much or as little as you want..."
            )
        
        with col2:
            st.subheader("Share Screenshots or Documents")
            uploaded_files = st.file_uploader(
                "Screenshots, chats, or any documents that help us understand",
                type=list(self.SUPPORTED_IMAGE_TYPES | self.SUPPORTED_TEXT_TYPES),
                accept_multiple_files=True,
                key="uploads",
            )
            
            image_files, uploaded_text_content = self.handle_file_uploads(uploaded_files)
        
        # Initial support button with more casual text
        if st.button("Talk to Your Recovery Squad 💝", type="primary"):
            self._handle_initial_support(user_input, uploaded_files, image_files, uploaded_text_content)

    def _handle_initial_support(self, user_input: str, uploaded_files, image_files, uploaded_text_content: str):
        """Handle initial support request"""
        if not st.session_state.api_key_input:
            st.warning("Please enter your API key in the sidebar first!")
        elif not st.session_state.agents_initialized:
            st.warning("Please wait for agents to initialize...")
        elif not (user_input or uploaded_files):
            st.warning("Tell us what's going on so we can help you!")
        else:
            self._provide_initial_support(user_input, image_files, uploaded_text_content)

    def _provide_initial_support(self, user_input: str, image_files, uploaded_text_content: str):
        """Provide initial support from all agents with human-like responses"""
        try:
            st.header("💬 Your Squad's Here For You")
            
            all_images = self.process_images(image_files)
            
            # Create more personal context like your original code
            context = f"Someone is going through a breakup and shared: '{user_input}'"
            if uploaded_text_content:
                context += f" They also shared some messages/documents: {uploaded_text_content[:500]}..."
            
            # Get responses from each agent with personalized prompts
            agents_data = [
                (st.session_state.therapist_agent, "💙 Maya says:", 
                 f"{context}\n\nRespond as Maya - be their caring friend who really gets what they're going through. Keep it warm, natural, and under 150 words."),
                (st.session_state.closure_agent, "✨ Alex suggests:", 
                 f"{context}\n\nHelp them find ONE specific way to get closure. Be gentle but direct. Under 100 words."),
                (st.session_state.routine_planner_agent, "🚀 Jordan's game plan:", 
                 f"{context}\n\nGive them 2-3 specific things they can do TODAY to start feeling better. Be encouraging and practical. Under 120 words."),
                (st.session_state.brutal_honesty_agent, "💪 Sam keeps it real:", 
                 f"{context}\n\nGive them the honest perspective they need to hear, but with love. Point out what they might be missing. Under 100 words.")
            ]
            
            for i, (agent, header, prompt) in enumerate(agents_data):
                if agent:
                    agent_name = header.split()[1]  # Extract name from header
                    with st.spinner(f"{agent_name} is responding..."):
                        try:
                            if i > 0:
                                time.sleep(self.RATE_LIMIT_DELAY)
                            
                            response = agent.run(message=prompt, images=all_images)
                            st.subheader(header)
                            st.markdown(response.content)
                            
                        except Exception as e:
                            error_msg = self._handle_agent_error(agent_name, e)
                            st.warning(f"{agent_name}: {error_msg}")
            
            st.markdown("---")
            st.markdown("💛 **You're stronger than you know. Take it one day at a time.**")
            
        except Exception as e:
            self.logger.error(f"Error during initial analysis: {str(e)}")
            st.error("Something went wrong. Please try again or check your API key.")

    def render_chat_section(self):
        """Render the ongoing chat interface"""
        st.markdown("---")
        st.header("💭 Continue Your Conversation")
        
        if not st.session_state.api_key_input:
            st.info("Enter your API key in the sidebar to start chatting with your recovery squad.")
        elif not st.session_state.agents_initialized:
            st.info("Initializing your recovery squad... Please wait a moment.")
        else:
            self._render_active_chat()

    def _render_active_chat(self):
        """Render the active chat interface"""
        # Display current chat setting
        if st.session_state.selected_agent == "All Squad Members":
            st.info("💬 **Group Chat**: All squad members will respond to your messages")
        else:
            config = self.AGENT_CONFIGS.get(st.session_state.selected_agent)
            if config:
                st.info(f"{config.emoji} **One-on-One**: Chatting with {st.session_state.selected_agent}")
        
        # Display chat history
        if st.session_state.chat_messages:
            st.subheader("Chat History")
            self._display_chat_history()
        
        # Chat input
        self._handle_chat_input()

    def _display_chat_history(self):
        """Display the chat message history"""
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        if message.get("agent"):
                            config = self.AGENT_CONFIGS.get(message["agent"])
                            emoji = config.emoji if config else "💬"
                            st.markdown(f"**{emoji} {message['agent']}:**")
                        st.markdown(message["content"])

    def _handle_chat_input(self):
        """Handle new chat input"""
        chat_input = st.chat_input("Continue your conversation with your recovery squad...")
        
        if chat_input:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": chat_input
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(chat_input)
            
            # Get and display agent responses
            with st.spinner("Your recovery squad is responding..."):
                contextual_message = self._build_contextual_message(chat_input)
                responses = self.get_agent_response(st.session_state.selected_agent, contextual_message)
                
                for agent_name, response in responses.items():
                    config = self.AGENT_CONFIGS.get(agent_name)
                    emoji = config.emoji if config else "💬"
                    
                    with st.chat_message("assistant"):
                        st.markdown(f"**{emoji} {agent_name}:**")
                        st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "agent": agent_name,
                        "content": response
                    })

    def _build_contextual_message(self, chat_input: str) -> str:
        """Build contextual message with chat history"""
        chat_history_context = ""
        if len(st.session_state.chat_messages) > 1:
            recent_messages = st.session_state.chat_messages[-6:]  # Last 3 exchanges
            chat_history_context = "\n\nRecent conversation context:\n"
            for msg in recent_messages[:-1]:  # Exclude current message
                role = "Human" if msg["role"] == "user" else f"{msg.get('agent', 'Assistant')}"
                chat_history_context += f"{role}: {msg['content'][:200]}...\n"
        
        return chat_input + chat_history_context

    def render_footer(self):
        """Render the application footer"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>💔➡️❤️ Made with love by Aamir's Recovery Squad</p>
            <p>Found this helpful? Share your recovery journey on "X" tag: @AmirKadri_7</p>
            <p>#breakup-agents</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show error logs if any
        if st.session_state.get("error_logs"):
            st.divider()
            st.subheader("Error Logs")
            for log_line in st.session_state.error_logs:
                st.text(log_line)

    def run(self):
        """Main application entry point"""
        self.render_sidebar()
        self.render_main_content()
        self.render_chat_section()
        self.render_footer()


# Application entry point
if __name__ == "__main__":
    app = RecoverySquadApp()
    app.run()