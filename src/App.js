import React, { useState } from "react";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("home");
  const [input, setInput] = useState("");

  return (
    <div className="page">

      {/*HEADER*/}
      <header className="header">
        <div 
          className={`tab ${activeTab === "hello" ? "active" : ""}`}
          onClick={() => setActiveTab("hello")}
        >
          Hello
        </div>
        <div 
          className={`tab ${activeTab === "home" ? "active" : ""}`}
          onClick={() => setActiveTab("home")}
        >
          Home Page
        </div>
        <div className="tab">About Us</div>
      </header>

      {/*MAIN CONTENT*/}
      <main className="content">

        {activeTab === "hello" && <h2>Hello Page</h2>}

        {activeTab === "home" && (
          <div>
            <h2>Type your password here</h2>

            {/*Input + Button Container*/}
            <div className="input-container">
              <input 
                className="input-box"
                type="text"
                value={input}
                placeholder="Enter password..."
                onChange={(e) => setInput(e.target.value)}
              />
              <button className="enter-btn">Enter</button>
            </div>

            <h2>Output</h2>

            {/*Output box*/}
            <div className="output-box"></div>
          </div>
        )}
      </main>

      {/*FOOTER*/}
      <footer className="footer">
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer">Facebook</a>
        <a href="https://twitter.com" target="_blank" rel="noopener noreferrer">Twitter</a>
        <a href="https://instagram.com" target="_blank" rel="noopener noreferrer">Instagram</a>
        <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">LinkedIn</a>
      </footer>

    </div>
  );
}

export default App;