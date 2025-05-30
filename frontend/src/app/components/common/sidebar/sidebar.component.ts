import { Component, EventEmitter, Output, OnInit } from '@angular/core';
import { HistoryService } from '../service/history.service';

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent implements OnInit {
  @Output() newChat = new EventEmitter<void>();
  @Output() sidebarToggled = new EventEmitter<boolean>(); // 👈 Emit sidebar state to parent

  isOpen = false;
  isMenuOpen = false;
  history: string[] = [];

  constructor(private historyService: HistoryService) {}

  ngOnInit() {
    this.historyService.history$.subscribe(hist => this.history = hist);
    this.historyService.fetchChatHistory();
  }

  toggleMenu() {
    this.isMenuOpen = !this.isMenuOpen;
  }

  toggleSidebar() {
    this.isOpen = !this.isOpen;
    this.sidebarToggled.emit(this.isOpen); // 👈 Inform parent component
  }

  onNewChatClick() {
    this.newChat.emit();
  }
}
