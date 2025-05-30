import { Component, ViewChild, ElementRef, AfterViewChecked, OnInit } from '@angular/core';
import { Chart, registerables } from 'chart.js';
import { HistoryService } from '../../common/service/history.service';
import { Subscription } from 'rxjs';

Chart.register(...registerables);

interface ChatMessage {
  text?: string;        // Now optional
  sender: 'user' | 'bot' | 'system';
  timestamp: Date;
  contentType?: 'chart' | 'table' | 'both' | 'text' | 'image';
  tableData?: any[][];
  tableHeaders?: string[];
  imageUrl?: string;    // URL for rendered images
  isSystemMessage?: boolean;
}


@Component({
  selector: 'app-mainpage',
  templateUrl: './mainpage.component.html',
  styleUrls: ['./mainpage.component.css']
})
export class MainpageComponent implements OnInit, AfterViewChecked {
  @ViewChild('scrollContainer') private scrollContainer!: ElementRef;
  @ViewChild('messageInput') private messageInput!: ElementRef;
isProcessing = false;
private currentSubscription: Subscription | null = null;
sidebarOpen = true; // Default to true if sidebar is open by default

onSidebarToggled(isOpen: boolean) {
  this.sidebarOpen = isOpen;
}
  question: string = '';
  loading = false; 
  history: string[] = [];
  isSystemMessage?: boolean; // ✅ Add this line

  lastQuestion: string = '';
  submitted: boolean = false;
  answerText: string = '';
questionType: 'chart' | 'table' | 'text' | 'both' | 'image' | null = null;

  chart: any;

  messages: ChatMessage[] = [];
  isTyping: boolean = false;
  darkMode: boolean = false;

  tableData = [
    { item: 'Item A', value: 10 },
    { item: 'Item B', value: 20 },
    { item: 'Item C', value: 30 }
  ];

  constructor(private historyService: HistoryService) {}

 ngOnInit() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    this.darkMode = true;
    this.applyTheme();
  }

  this.historyService.history$.subscribe(history => {
    this.history = history;
  });

  this.historyService.fetchChatHistory(); // Load chat history from backend

  setTimeout(() => {
    this.addMessage('Hi there! 👋 How can I help you today?', 'bot');
    this.submitted = true;
  }, 500);
}


  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  scrollToBottom(): void {
    try {
      this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
    } catch (err) {}
  }

  toggleTheme(): void {
    this.darkMode = !this.darkMode;
    this.applyTheme();
    localStorage.setItem('theme', this.darkMode ? 'dark' : 'light');
  }

  applyTheme(): void {
    document.body.classList.toggle('dark-theme', this.darkMode);
  }

  onEnterPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault?.();
      this.onSubmit();
    }
  }

  addMessage(text: string, sender: 'user' | 'bot', contentType?: 'chart' | 'table' | 'both' | 'text'): void {
    const message: ChatMessage = {
      text,
      sender,
      timestamp: new Date()
    };

    if (contentType) {
      message.contentType = contentType;

      if (contentType === 'table' || contentType === 'both') {
        message.tableData = this.formatTableData();
        message.tableHeaders = ['Item', 'Value'];
      }
    }

    this.messages.push(message);
  }

  formatTableData(): any[] {
    return this.tableData.map(item => [item.item, item.value]);
  }
onSubmit() {
  const trimmed = this.question.trim();
  if (!trimmed) return;

  this.addMessage(trimmed, 'user');
  this.submitted = true;
  this.historyService.addQuestion(trimmed);
  this.lastQuestion = trimmed;

  if (this.messageInput) {
    this.messageInput.nativeElement.style.height = 'auto';
  }

  this.question = '';
  this.isTyping = true;
  this.isProcessing = true;

  this.processQuestion(trimmed).add(() => {
    this.isTyping = false;
    this.isProcessing = false;
  });
}
cancelSubmission() {
  this.isTyping = false;
  this.isProcessing = false;

  if (this.currentSubscription) {
    this.currentSubscription.unsubscribe();
    this.currentSubscription = null;
  }

  const message: ChatMessage = {
    text: 'Message cancelled.',
    sender: 'system',
    timestamp: new Date(),
    contentType: 'text',
    isSystemMessage: true
  };

  this.messages.push(message);
}

//   onSubmit() {
//   const trimmed = this.question.trim();
//   if (!trimmed) return;

//   this.addMessage(trimmed, 'user');
//   this.submitted = true;
//   this.historyService.addQuestion(trimmed);
//   this.lastQuestion = trimmed;

//   if (this.messageInput) {
//     this.messageInput.nativeElement.style.height = 'auto';
//   }

//   this.question = '';
//   this.isTyping = true;

//   this.processQuestion(trimmed).add(() => {
//     this.isTyping = false;
//   });
// }
startNewChat(): void {
  this.messages = [];             // Clears current chat messages
  this.question = '';             // Clears the input box
  this.answerText = '';           // Clears any current answer text
  this.questionType = null;       // Resets the type
  this.lastQuestion = '';         // Clears last question
  this.submitted = false;
  this.isTyping = false;

  // Re-add greeting
  setTimeout(() => {
    this.addMessage('Hi there! 👋 How can I help you today?', 'bot');
    this.submitted = true;
  }, 100);
}
processQuestion(question: string) {
  if (this.currentSubscription) {
    this.currentSubscription.unsubscribe();
  }

  this.currentSubscription = this.historyService
    .sendQuestionToAPI(question)
    .subscribe(
      (response) => {
        const explanation = response?.Explanation || 'Here is the response.';
        const table = response?.Table;
        const graph = response?.Graph;
        const graphImage = response?.GraphImage;

        // 1) Always show the explanation text first
        this.messages.push({
          text: explanation,
          sender: 'bot',
          timestamp: new Date(),
          contentType: 'text'
        });

        // 2) If there’s a table, push it
        if (Array.isArray(table) && table.length > 0) {
          const headers = Object.keys(table[0]);
          const formattedData = table.map((row: any) =>
            headers.map((key) => row[key])
          );

          this.messages.push({
            sender: 'bot',
            timestamp: new Date(),
            contentType: 'table',
            tableHeaders: headers,
            tableData: formattedData
          });
        }

        // 3) If there’s an image, push it
        if (graphImage && graphImage !== 'NA') {
          const fullImageUrl = `http://127.0.0.1:8000${graphImage}`;
          console.log('Image URL:', fullImageUrl);

          this.messages.push({
            sender: 'bot',
            timestamp: new Date(),
            contentType: 'image',
            imageUrl: fullImageUrl
          });
        }
        // done processing
        this.isTyping = false;
        this.isProcessing = false;
      },
      (error) => {
        console.error('API error:', error);
        this.messages.push({
          text: 'Sorry, there was an error fetching the data.',
          sender: 'bot',
          timestamp: new Date(),
          contentType: 'text'
        });
        this.isTyping = false;
        this.isProcessing = false;
      }
    );

  return this.currentSubscription;
}



  // processQuestion(question: string) {
  //   const lower = question.toLowerCase();

  //   if (lower === 'show me chart and table') {
  //     this.answerText = 'Here are both a chart and a table based on your request.';
  //     this.questionType = 'both';
  //     this.addMessage(this.answerText, 'bot', 'both');
  //     setTimeout(() => this.renderChart(), 100);
  //   } else if (lower === 'show me a chart') {
  //     this.answerText = 'Here is a sample chart based on your question.';
  //     this.questionType = 'chart';
  //     this.addMessage(this.answerText, 'bot', 'chart');
  //     setTimeout(() => this.renderChart(), 100);
  //   } else if (lower === 'show me a table') {
  //     this.answerText = 'Here is a sample table based on your question.';
  //     this.questionType = 'table';
  //     this.addMessage(this.answerText, 'bot', 'table');
  //   } else {
  //     this.answerText = `You asked: "${question}" — Here's a simple answer.`;
  //     this.questionType = 'text';
  //     this.addMessage(this.answerText, 'bot', 'text');
  //   }
  // }

  renderChart() {
    if (this.chart) {
      this.chart.destroy();
    }

    const ctx = document.getElementById('myChart') as HTMLCanvasElement;
    if (ctx) {
      this.chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['Jan', 'Feb', 'Mar', 'Apr'],
          datasets: [{
            label: 'Dummy Data',
            data: [10, 20, 30, 25],
            borderColor: '#4a6fa5',
            backgroundColor: 'rgba(74, 111, 165, 0.2)',
            borderWidth: 2,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: true },
            tooltip: { mode: 'index', intersect: false }
          },
          hover: { mode: 'nearest', intersect: true },
          scales: {
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(0, 0, 0, 0.05)'
              }
            },
            x: {
              grid: {
                display: false
              }
            }
          }
        }
      });
    }
  }
}
